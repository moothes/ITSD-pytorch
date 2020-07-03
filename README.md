# ITSD-pytorch
Code for CVPR 2020 [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf) "Interactive Two-Stream Decoder for Accurate and Fast Saliency Detection"

Saliency maps can be download at: VGG ([Baidu Yun](https://pan.baidu.com/s/1AdkLgfOK1jwgcqk06zwOwQ) \[gf1i\]), Resnet ([Baidu Yun](https://pan.baidu.com/s/1Gu9RpKuMdZrj1iJvh4A2og) \[sanf\])

## Prerequisites

- [Pytorch 1.0.0](http://pytorch.org/)
- [torchvision 0.2.1](http://pytorch.org/)
- Thop
- Progress

## Usage：
Imagenet-pretrained weights can be download at [Pytorch Models](https://pytorch.org/hub/research-models).

Or using our own implement: VGG ([Baidu Yun](https://pan.baidu.com/s/1Ii1Z3qqCxSk9LB6tiA9Q1g) \[xkxh\]), Resnet ([Baidu Yun](https://pan.baidu.com/s/1_-A3ACWKZEN1VXtKTAo3Nw) \[rc2n\]).

Please refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).
 
### Training：
```bash
python3 train.py --sub=[job_name] --ids=[gpus] --model=[vgg/resnet]
```

### Testing:
```bash
python3 test.py --sub=[job_name] --ids=[gpus] --model=[vgg/resnet]
```

## Contact
If you have any questions, feel free to contact me via: `mootheszhou@gmail.com`.


## Bibtex
```latex
@InProceedings{Zhou_2020_CVPR,
author = {Zhou, Huajun and Xie, Xiaohua and Lai, Jian-Huang and Chen, Zixuan and Yang, Lingxiao},
title = {Interactive Two-Stream Decoder for Accurate and Fast Saliency Detection},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
} 
```
