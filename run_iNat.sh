#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
# model=conv4.config
# python ./src/train_inatural_lshot.py -c ./configs/inatural/softmax/$model --data ./data/iNat/setup --log-file /simpleshot.log --evaluate
lmd=1.0
# ##
 ## Resnet 10
model=resnet10.config
python ./src/train_inatural_lshot.py -c ./configs/inatural/softmax/$model --lmd $lmd --data ./data/iNat/setup/ --lshot --knn 10 --log-file /LaplacianShot.log --evaluate
 ##
# ## Resnet 18
model=resnet18.config
python ./src/train_inatural_lshot.py -c ./configs/inatural/softmax/$model --lmd $lmd --data ./data/iNat/setup/ --lshot --knn 10 --log-file /LaplacianShot.log --evaluate

 ## Resnet 50
model=resnet50.config
python ./src/train_inatural_lshot.py -c ./configs/inatural/softmax/$model --lmd $lmd --data ./data/iNat/setup/ --lshot --knn 10 --log-file /LaplacianShot.log --evaluate
## WRN
model=wideres.config
python ./src/train_inatural_lshot.py -c ./configs/inatural/softmax/$model --lmd $lmd --data ./data/iNat/setup/ --lshot --knn 10 --log-file /LaplacianShot.log --evaluate
# ### Mobilenet
model=mobilenet.config
python ./src/train_inatural_lshot.py -c ./configs/inatural/softmax/$model --lmd $lmd --data ./data/iNat/setup/ --lshot --knn 10 --log-file /LaplacianShot.log --evaluate
# ### Densenet
model=densenet121.config
python ./src/train_inatural_lshot.py -c ./configs/inatural/softmax/$model --lmd $lmd --data ./data/iNat/setup/ --lshot --knn 10 --log-file /LaplacianShot.log --evaluate