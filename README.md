# Introduction

Official PyTorch implementation for L-MSFC: [End-to-End Learnable Multi-Scale Feature Compression](https://ieeexplore.ieee.org/abstract/document/10210338), IEEE TCSVT 2023 (Early Access).

In this repository, we provide the source code for the proposed models and training scripts, but not for evaluation scripts.

# Requirements

- ### PyTorch

  Install PyTorch package compatible with your cuda version like:

  ```
  pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
  ```

- ### CompressAI

  L-MSFC consists of FENet (Fusion and Encoding), DRNet (Decoding and Reconstruction), and shared entropy model.

  For the shared entropy model, we have two options:

  - `JointAutoregressiveHierarchicalPriors` class from compressAI (w/ CM)
  - `MeanScaleHyperprior` class from compressAI (w/o CM)

  Install compressai with pip:

  ```
  pip install compressai==1.1.5
  ```

- ### Detectron2

  We used two pretrained feature extractors from Detectron2 as follows:

  - Object detection: `COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x`
  - Instance segmentation: `COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x`

  Install detectron2 following instructions from the [link](https://github.com/facebookresearch/detectron2/releases/tag/v0.4) or run:

  ```
   python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
  ```

- ### Other Dependencies
  ```
  pillow==9.5.0
  tensorboardX
  opencv-python
  ```

# Training

- ### Dataset

  We randomly sampled approximately 90k images from [OpenImagesV6 dataset](https://storage.googleapis.com/openimages/web/index.html) and used 100 of these images for validation.

- ### Model

  There are two models available: L-MSFC and L-MSFC (w/o CM), located in './src/model.py' and './src/model_no_ar.py', respectively.

  Each model can be trained with a quality factor ranging from 1 to 6.

  You can select the vision task (feature extractor) you wish to use from the options: ['detection', 'segmentation'].

- ### Example
  For example, run:
  ```
  CUDA_VISIBLE_DEVICES={gpu_id} python train.py --model model_no_ar --quality 3 --task detection --batch-size 1 --patch-size 0 --dataset ./data/ --savedir ./save/no_ar_det_q3 --logdir ./log/no_ar_det_q3
  ```

# Pretrained Checkpoints

[Google drive link](https://drive.google.com/drive/folders/1I7gau7tfsneBlKDcCKQr5cVqp-cn9oKM?usp=sharing) (Available from October 31, 2023)

# Acknowledgement

The implementation is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI)

# Citation

If this work is useful for your research, please cite:

```
@ARTICLE{10210338,
  author={Kim, Yeongwoong and Jeong, Hyewon and Yu, Janghyun and Kim, Younhee and Lee, Jooyoung and Jeong, Se Yoon and Kim, Hui Yong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={End-to-End Learnable Multi-Scale Feature Compression for VCM}, 
  year={2024},
  volume={34},
  number={5},
  pages={3156-3167},
  keywords={Feature extraction;Image coding;Task analysis;Transform coding;Decoding;Image reconstruction;Encoding;Video coding for machine (VCM);feature compression;learned image compression;versatile video coding (VVC)},
  doi={10.1109/TCSVT.2023.3302858}}
```
