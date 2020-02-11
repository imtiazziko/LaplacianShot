# LaplacianShot for few shot learning

This repository contains the code for **LaplacianShot**. The code is based on the [SimpleShot github](https://github.com/mileyan/simple_shot)


## Introduction
We propose LaplacianShot for few-shot tasks, which integrates two types of potentials: (1) unary potentials assigning query samples to the nearest class prototype, and (2) pairwise Laplacian potentials encouraging nearby query samples to have consistent predictions. Our algorithm is performed during inference in few-shot scenarios, following the traditional training of a deep convolutional network on the base classes with the standard cross-entropy loss.

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
We follow the instruction from https://github.com/daviswer/fewshotlocal. Download and unpack the iNat2017 _Training and validation images_, and the _Training bounding box annotations_, to [data/iNat](./data/iNat) directory from [here](https://github.com/visipedia/inat_comp/blob/master/2017/README.md#Data).
 Create the meta-iNat dataset by running:
 ```angular2
cd ./data/iNat
python iNat_setup.py
```

And run the following script to generate split files.
```angular2
python ./src/inatural_split.py --data path-to-inat/setup --split ./split/inatural/
```

### 3 Train and Test
You can download the pretrained convolutional network models on base classes by running:
```angular2
cd ./src
python download_models.py
```

you can run the following scripts to test and run the code.

In miniImageNet:
```angular2
sh run_mini.sh
```
In tieredImageNet
```angular2
sh run_tiered.sh
```
In iNat
```angular2
sh run_iNat.sh
```

