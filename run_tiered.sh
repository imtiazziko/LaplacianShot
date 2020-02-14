#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
# train on tiered imagenet
#Simpleshot
#model=resnet10.config
# python ./src/train_lshot.py -c ./configs/tiered/softmax/$model --data ./data/tiered-imagenet/data/ --log-file /simpleshot.log --evaluate

tune=True
lmd=1.0

# ## Resnet 10
model=resnet10.config
python ./src/train_lshot.py -c ./configs/tiered/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/tiered-imagenet/data/ --lshot  --log-file /LaplacianShot.log --evaluate
# ##
# ## Resnet 18
model=resnet18.config
python ./src/train_lshot.py -c ./configs/tiered/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/tiered-imagenet/data/ --lshot  --log-file /LaplacianShot.log --evaluate
# ##
### WRN
model=wideres.config
python ./src/train_lshot.py -c ./configs/tiered/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/tiered-imagenet/data/ --lshot  --log-file /LaplacianShot.log --evaluate
# ### Mobilenet
model=mobilenet.config
python ./src/train_lshot.py -c ./configs/tiered/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/tiered-imagenet/data/ --lshot  --log-file /LaplacianShot.log --evaluate
# ### Densenet
model=densenet121.config
python ./src/train_lshot.py -c ./configs/tiered/softmax/$model --lmd $lmd --tune-lmd $tune  --data ./data/tiered-imagenet/data/ --lshot  --log-file /LaplacianShot.log --evaluate
