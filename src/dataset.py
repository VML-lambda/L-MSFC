from pathlib import Path

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OpenImageDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        # Openimage
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = cv2.imread(str(self.samples[index]))
        if self.transform is not None:
            img, _ = T.apply_transform_gens(self.transform, img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return img

    def __len__(self):
        return len(self.samples)


class DF2KDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        # Openimage
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = cv2.imread(str(self.samples[index]))
        if self.transform is not None:
            img, _ = T.apply_transform_gens(self.transform, img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return img

    def __len__(self):
        return len(self.samples)


def toImgPIL(imgOpenCV):
    return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB))


def toImgOpenCV(imgPIL):  # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL)  # After mapping from PIL to numpy : [R,G,B,A]
    # numpy Image Channel system: [B,G,R,A]
    red = i[:, :, 0].copy()
    i[:, :, 0] = i[:, :, 2].copy()
    i[:, :, 2] = red
    return i
