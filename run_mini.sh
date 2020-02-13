#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
# train on mini imagenet
#Simpleshot
#model=resnet10.config
# python ./src/train_lshot.py -c ./configs/mini/softmax/$model --data ./data/images --log-file /simpleshot.log --evaluate

tune=True
lmd=1.0 # Overridden when tune=True

# ## Resnet 10
model=resnet10.config
python ./src/train_lshot.py -c ./configs/mini/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/images --lshot --log-file /LaplacianShot.log --evaluate --enlarge
# ##
# ## Resnet 18
model=resnet18.config
python ./src/train_lshot.py -c ./configs/mini/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/images --lshot  --log-file /LaplacianShot.log --evaluate --enlarge
### WRN
model=wideres.config
python ./src/train_lshot.py -c ./configs/mini/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/images --lshot  --log-file /LaplacianShot.log --evaluate --enlarge
# ### Mobilenet
model=mobilenet.config
python ./src/train_lshot.py -c ./configs/mini/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/images --lshot  --log-file /LaplacianShot.log --evaluate --enlarge
# ### Densenet
model=densenet121.config
python ./src/train_lshot.py -c ./configs/mini/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/images --lshot  --log-file /LaplacianShot.log --evaluate --enlarge

