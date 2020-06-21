# ITSD-pytorch
Code for CVPR 2020 paper "Interactive Two-Stream Decoder for Accurate and Fast Saliency Detection"

Saliency maps can be download at: VGG ([Baidu Yun](https://pan.baidu.com/s/1AdkLgfOK1jwgcqk06zwOwQ) \[gf1i\]), Resnet ([Baidu Yun](https://pan.baidu.com/s/1Gu9RpKuMdZrj1iJvh4A2og) \[sanf\])

### Usage：
Imagenet-pretrained weights: VGG ([Baidu Yun](https://pan.baidu.com/s/1Ii1Z3qqCxSk9LB6tiA9Q1g) \[xkxh\]), Resnet ([Baidu Yun](https://pan.baidu.com/s/1_-A3ACWKZEN1VXtKTAo3Nw) \[rc2n\])
 
Training：
```bash
python3 train.py --sub=[job_name] --ids=[gpus] 
```

Testing:
```bash
python3 test.py --sub=[job_name] --ids=[gpus] 
```

### Notice: 
Our evaluation code is slightly lower than [public evaluation](https://github.com/Andrew-Qibin/SalMetric). Please use this code to get the results in our paper. 

