#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

tune=True
lmd=1.0 # Overridden when tune=True , except for iNat
protrec=True # Add query to compute prototoypes. Set False to compute with support examples only
datapath=./data/images  # for mini
#datapath=./data/tiered-imagenet/data  # for tiered
#datapath=./data/CUB/CUB_200_2011/images # for CUB

## Change the network name accordingly
config=./configs/mini/softmax/resnet18.config # mini
#config=./configs/tiered/softmax/resnet18.config # tiered
#config=./configs/cub/softmax/resnet18.config # cub
#
python ./src/train_lshot.py -c $config --proto-rect $protrec --lmd $lmd --tune-lmd $tune  --data $datapath --lshot --log-file /LaplacianShot.log --evaluate

## iNat
#datapath=./data/iNat/setup/ # iNat
#config=./configs/inatural/softmax/resnet18.config # iNat
#python ./src/train_inatural_lshot.py -c $config --lmd $lmd --data $datapath --lshot --log-file /LaplacianShot.log --evaluate
