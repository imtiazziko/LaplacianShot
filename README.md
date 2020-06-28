# LaplacianShot: Laplacian Regularized Few Shot Learning

This repository contains the code for **LaplacianShot**. The code is adapted from [SimpleShot github](https://github.com/mileyan/simple_shot).

If you use this code please cite the following ICML 2020 paper:

[**Laplacian Regularized Few-shot Learning**]()  
Imtiaz Masud Ziko, Jose Dolz, Eric Granger and Ismail Ben Ayed  
In ICML 2020.

## Introduction
We propose LaplacianShot for few-shot learning tasks, which integrates two types of potentials: (1) unary potentials assigning query samples to the nearest class prototype, and (2) pairwise Laplacian potentials encouraging nearby query samples to have consistent predictions. 

LaplacianShot is utilized during inference in few-shot scenarios, following the traditional training of a deep convolutional network on the base classes with the cross-entropy loss. In fact, LaplacianShot can be used with any **learned feature extractor during inference**.

## Usage
### 1. Dependencies
- Python 3.6+
- Pytorch 1.0+

### 2. Datasets
#### 2.1 Mini-ImageNet
You can download the dataset from [here](https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE)

#### 2.2 Tiered-ImageNet
You can download the Tiered-ImageNet from [here](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view).
After downloading and unziping this dataset run the following script to generate split files.
```angular2
python src/utils/tieredImagenet.py --data path-to-tiered --split split/tiered/
```
#### 2.3 CUB
Download and unpack the CUB 200-2011 from [here](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
After downloading and unziping this dataset run the following script to generate split files.
```angular2
python src/utils/cub.py --data path-to-cub --split split/cub/
```
#### 2.4 iNat2017
We follow the instruction from https://github.com/daviswer/fewshotlocal. Download and unpack the iNat2017 _Training and validation images_, and the _Training bounding box annotations_, to [data/iNat](./data/iNat) directory from [here](https://github.com/visipedia/inat_comp/blob/master/2017/README.md#Data). Also download _traincatlist.pth_ and _testcatlist.pth_ in the same directory from [here](https://github.com/daviswer/fewshotlocal) and finally
 create the meta-iNat dataset by running:
 ```angular2
cd ./data/iNat
python iNat_setup.py
```

And run the following script to generate split files.
```angular2
python ./src/inatural_split.py --data path-to-inat/setup --split ./split/inatural/
```

### 3 Train and Test
You can download our pretrained network models on base classes by running:
```angular2
cd ./src
python download_models.py
```
Alternatively to train the network on the base classes from scratch remove the "--evaluate " options in the following script.
The scripts to test LaplacianShot:
```angular2
sh run.sh
```
You can change the commented options accordingly for each dataset.

Some of our results of LaplacianShot with WRN network on mini/tiered-ImageNet and CUB dataset:

| Dataset | Network   | 1-shot | 5-shot |
|---------|-----------|--------|--------|
| miniImageNet    | WRN       | 74.86  | 84.13  |
| tieredImageNet  | WRN       | 80.18  | 87.56  |
| CUB     | ResNet-18 | 80.96  | 88.68  |