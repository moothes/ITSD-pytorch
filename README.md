# ITSD-pytorch
Code for CVPR 2020 [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Interactive_Two-Stream_Decoder_for_Accurate_and_Fast_Saliency_Detection_CVPR_2020_paper.pdf) "Interactive Two-Stream Decoder for Accurate and Fast Saliency Detection"

Saliency maps can be download at: VGG ([Baidu Yun](https://pan.baidu.com/s/1AdkLgfOK1jwgcqk06zwOwQ) \[gf1i\]), Resnet ([Baidu Yun](https://pan.baidu.com/s/1Gu9RpKuMdZrj1iJvh4A2og) \[sanf\])

## We have prepared a gift for the coming new year! Available soon!

## Prerequisites

- [Pytorch 1.0.0](http://pytorch.org/)
- [torchvision 0.2.1](http://pytorch.org/)
- Thop
- Progress

## Usage：
Official imagenet-pretrained weights can be download at [Resnet50](https://download.pytorch.org/models/resnet50-19c8e357.pth) and [VGG16](https://download.pytorch.org/models/vgg16-397923af.pth).

Our models: Google drive ([ResNet50](https://drive.google.com/file/d/1qcZOOL7b7DJ0VbtXK0MvDsOTGNSsmXDm/view) and [VGG16](https://drive.google.com/file/d/1zyDqrjIacqK83pyzbq90rys9m732n28j/view)) or Baidu Disk ([Resnet50](https://pan.baidu.com/s/1qKSnPqbNs4--PwB5fA4E-g) [y55w] and [VGG16](https://pan.baidu.com/s/1ceI8lReLozh2WRsylszQgA) [kehh])

Please refer to this repo for results evaluation: [SalMetric](https://github.com/Andrew-Qibin/SalMetric).
 
### Training：
```bash
python3 train.py --sub=[job_name] --ids=[gpus] --model=[vgg/resnet]
```

### Testing:
```bash

mv path_to_model ./save/[vgg/resnet]/[job_name]/final.pkl  # if testing the provided models
python3 test.py --sub=[job_name] --ids=[gpus] --model=[vgg/resnet]
```


## Contact
If you have any question, feel free to contact me via: `mootheszhou@gmail.com`.


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
