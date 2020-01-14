# Laplacian few shot learning

This repository contains the code for Laplacian few shot learning. This code is based on the SimpleShot github [https://github.com/mileyan/simple_shot](https://github.com/mileyan/simple_shot)

[Laplacian few shot learning]()

by Imtiaz Masud Ziko and Ismail Ben Ayed

## Citation
If you find Simple Shot useful in your research, please consider citing:
```angular2
@article{ziko2019,
  title={Laplacian few shot learning},
  author={Imtiaz Ziko and Ismail Ben Ayed},
  journal={arXiv preprint },
  year={2019}
}
```

## Introduction
Recent SimpleShot paper showed simple feature transformations suffice to obtain
competitive few-shot learning accuracies using simple nearest-neighbor rules in combination with mean-subtraction
and L2-normalization and outperforms prior results in three out of five settings
on the miniImageNet dataset.

## Usage
### 1. Dependencies
- Python 3.5+
- Pytorch 1.0+

### 2. Download Datasets
### 2.1 Mini-ImageNet
You can download the dataset from https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

### 2.2 Tiered-ImageNet
You can download the dataset from https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view.
After downloading and unziping this dataset, you have to run the follow script to generate split files.
```angular2
python src/utils/tieredImagenet.py --data path-to-tiered --split split/tiered/
```
### 2.3 iNat2017
Please follow the instruction from https://github.com/daviswer/fewshotlocal to download the dataset.
And run the following script to generate split files.
```angular2
python ./src/inatural_split.py --data path-to-inat/setup --split ./split/inatural/
```

### 3 Train and Test
You can download the pretrained models from:

Google Drives: https://drive.google.com/open?id=14ZCz3l11ehCl8_E1P0YSbF__PK4SwcBZ

BaiduYun: https://pan.baidu.com/s/1tC2IU1JBL5vPNmnxXMu2sA  code:d3j5

Or, you can download them by running
```angular2
cd ./src
python download_models.py
```
This repo includes `Resnet-10/18/34/50`, `Densenet-121`, `Conv-4`, `WRN`, `MobileNet` models.
For instance, If you would like to train a Conv-4 on Mini-ImageNet, you can run
```angular2
python ./src/train.py -c ./configs/mini/softmax/conv4.config
```
The evaluation command of mini/tiered-imagenet is
```angular2
python ./src/train.py -c ./configs/mini/softmax/conv4.config --evaluate --enlarge
```
To evaluate INat models,
```angular2
python ./src/test_inatural.py -c ./configs/inatural/softmax/conv4.config --evaluate --enlarge
```
## Contact
If you have any question, please feel free to email us.

Yan Wang (yw763@cornell.edu)

