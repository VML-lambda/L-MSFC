import time
import torch
import torch.nn.functional as F
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from compressai.models import JointAutoregressiveHierarchicalPriors
from torch import nn

from src.utils.stream_helper import *


class FeatureCombine(nn.Module):
    def __init__(self, N, M) -> None:
        super().__init__()
        self.p2Encoder = nn.Sequential(
            ResidualBlockWithStride(N, M, stride=2),
            ResidualBlock(M, M),
        )

        self.p3Encoder = nn.Sequential(
            ResidualBlockWithStride(N + M, M, stride=2),
            AttentionBlock(M),
            ResidualBlock(M, M),
        )

        self.p4Encoder = nn.Sequential(
            ResidualBlockWithStride(N + M, M, stride=2),
            ResidualBlock(M, M),
        )

        self.p5Encoder = nn.Sequential(
            conv3x3(N + M, M, stride=2),
            AttentionBlock(M),
        )

    def forward(self, p_layer_features):
        # p_layer_features contains padded features p2, p3, p4, p5
        p2, p3, p4, p5 = tuple(p_layer_features)
        y = self.p2Encoder(p2)
        y = self.p3Encoder(torch.cat([y, p3], dim=1))
        y = self.p4Encoder(torch.cat([y, p4], dim=1))
        y = self.p5Encoder(torch.cat([y, p5], dim=1))
        return y


class FeatureSynthesis(nn.Module):
    def __init__(self, N, M) -> None:
        super().__init__()

        class FeatureMixingBlock(nn.Module):
            def __init__(self, N) -> None:
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(N * 2, N, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                )

            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low

        self.p5Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )

        self.p4Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )

        self.p3Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            AttentionBlock(M),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, N, 2),
        )
        self.p2Decoder = nn.Sequential(
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            AttentionBlock(M),
            ResidualBlock(M, M),
            ResidualBlockUpsample(M, M, 2),
            ResidualBlock(M, M),
            subpel_conv3x3(M, N, 2),
        )

        self.decoder_attention = AttentionBlock(M)

        self.fmb23 = FeatureMixingBlock(N)
        self.fmb34 = FeatureMixingBlock(N)
        self.fmb45 = FeatureMixingBlock(N)

    def forward(self, y_hat):
        y_hat = self.decoder_attention(y_hat)
        p2 = self.p2Decoder(y_hat)
        p3 = self.fmb23(p2, self.p3Decoder(y_hat))
        p4 = self.fmb34(p3, self.p4Decoder(y_hat))
        p5 = self.fmb45(p4, self.p5Decoder(y_hat))
        return [p2, p3, p4, p5]


class FeatureCompressor(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=256, M=128, **kwargs):
        super().__init__(M, M, **kwargs)

        self.g_a = FeatureCombine(N, M)
        self.g_s = FeatureSynthesis(N, M)

        self.h_a = nn.Sequential(
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(M, M),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(M, M, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(M * 3 // 2, M * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(M * 3 // 2, M * 2),
        )

        self.p6Decoder = nn.Sequential(nn.MaxPool2d(1, stride=2))

    def forward(self, features):  # features: [p2, p3, p4, p5, p6]
        features = features[:-1]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        features = self.feature_padding(features, pad_info)

        y = self.g_a(features)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        recon_p_layer_features = self.g_s(y_hat)
        recon_p_layer_features = self.feature_unpadding(
            recon_p_layer_features, pad_info
        )

        p6 = self.p6Decoder(
            recon_p_layer_features[3]
        )  # p6 is generated from p5 directly

        recon_p_layer_features.append(p6)

        return {
            "features": recon_p_layer_features,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, features):  # features: [p2, p3, p4, p5, p6]
        features = features[:-1]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        features = self.feature_padding(features, pad_info)
        y = self.g_a(features)

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, p2_h, p2_w):
        assert isinstance(strings, list) and len(strings) == 2
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        padded_p2_h = pad_info["padded_size"][0][0]
        padded_p2_w = pad_info["padded_size"][0][1]
        z_shape = get_downsampled_shape(padded_p2_h, padded_p2_w, 64)

        z_hat = self.entropy_bottleneck.decompress(strings[1], z_shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))

        recon_p_layer_features = self.g_s(y_hat)
        recon_p_layer_features = self.feature_unpadding(
            recon_p_layer_features, pad_info
        )
        p6 = self.p6Decoder(
            recon_p_layer_features[3]
        )  # p6 is generated from p5 directly
        recon_p_layer_features.append(p6)
        return {"features": recon_p_layer_features, "y_hat": y_hat}

    def encode_decode(self, features, output_path, p2_height, p2_width):
        encoding_time_start = time.time()
        encoded = self.encode(features, output_path, p2_height, p2_width)
        encoding_time = time.time() - encoding_time_start
        decoding_time_start = time.time()
        decoded = self.decode(output_path)
        decoding_time = time.time() - decoding_time_start
        encoded.update(decoded)
        encoded["encoding_time"] = encoding_time
        encoded["decoding_time"] = decoding_time
        return encoded

    def encode(self, features, output_path, p2_height, p2_width):
        encoded = self.compress(features)
        y_string = encoded["strings"][0][0]
        z_string = encoded["strings"][1][0]

        encode_feature(p2_height, p2_width, y_string, z_string, output_path)
        bits = filesize(output_path) * 8
        summary = {
            "bit": bits,
            "bit_y": len(y_string) * 8,
            "bit_z": len(z_string) * 8,
        }
        encoded.update(summary)
        return encoded

    def decode(self, input_path):
        p2_height, p2_width, y_string, z_string = decode_feature(input_path)
        decoded = self.decompress([y_string, z_string], p2_height, p2_width)
        return decoded

    def cal_feature_padding_size(self, p2_shape):
        ps_list = [64, 32, 16, 8]
        ori_size = []
        paddings = []
        unpaddings = []
        padded_size = []

        ori_size.append(p2_shape)
        for i in range(len(ps_list) - 1):
            h, w = ori_size[-1]
            ori_size.append(((h + 1) // 2, (w + 1) // 2))

        for i, ps in enumerate(ps_list):
            h = ori_size[i][0]
            w = ori_size[i][1]

            h_pad_len = ps - h % ps if h % ps != 0 else 0
            w_pad_len = ps - w % ps if w % ps != 0 else 0

            paddings.append(
                (
                    w_pad_len // 2,
                    w_pad_len - w_pad_len // 2,
                    h_pad_len // 2,
                    h_pad_len - h_pad_len // 2,
                )
            )
            unpaddings.append(
                (
                    0 - (w_pad_len // 2),
                    0 - (w_pad_len - w_pad_len // 2),
                    0 - (h_pad_len // 2),
                    0 - (h_pad_len - h_pad_len // 2),
                )
            )

        for i, p in enumerate(paddings):
            h = ori_size[i][0]
            w = ori_size[i][1]
            h_pad_len = p[2] + p[3]
            w_pad_len = p[0] + p[1]
            padded_size.append((h + h_pad_len, w + w_pad_len))

        return {
            "ori_size": ori_size,
            "paddings": paddings,
            "unpaddings": unpaddings,
            "padded_size": padded_size,
        }

    def feature_padding(self, features, pad_info):
        p2, p3, p4, p5 = features
        paddings = pad_info["paddings"]

        p2 = F.pad(p2, paddings[0], mode="reflect")
        p3 = F.pad(p3, paddings[1], mode="reflect")
        p4 = F.pad(p4, paddings[2], mode="reflect")
        p5 = F.pad(p5, paddings[3], mode="reflect")
        return [p2, p3, p4, p5]

    def feature_unpadding(self, features, pad_info):
        p2, p3, p4, p5 = features
        unpaddings = pad_info["unpaddings"]

        p2 = F.pad(p2, unpaddings[0])
        p3 = F.pad(p3, unpaddings[1])
        p4 = F.pad(p4, unpaddings[2])
        p5 = F.pad(p5, unpaddings[3])
        return [p2, p3, p4, p5]
