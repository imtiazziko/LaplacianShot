#!/bin/bash
#cd ./src
#python download_models.py
#python ./src/train.py -c ./configs/tiered/softmax/resnet18.config --data ./data/tiered-imagenet/data
#python ./src/train_slk.py -c ./configs/tiered/softmax/resnet18.config --data ./data/tiered-imagenet/data --slk --lmd 0.8 --knn 3 --log-file '/slk-knn-3-lambda-0.8-2-test.log'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# train on tiered imagenet
enlarge=True
# query=15
tune=True
# conv
# model=conv4.config
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge
# ##
# ## Resnet 10
# model=resnet10.config
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

# ## Resnet 18
# model=resnet18.config
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

# ### WRN
# model=wideres.config
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

# ### Mobilenet
# model=mobilenet.config
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

# ### Densenet
# model=densenet121.config
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

##query 200
query=215
# conv
model=conv4.config
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge
##
## Resnet 10
model=resnet10.config
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

## Resnet 18
model=resnet18.config
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

### WRN
model=wideres.config
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

### Mobilenet
model=mobilenet.config
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge

### Densenet
model=densenet121.config
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3.log --evaluate --enlarge $enlarge
# python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 1 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation_original.py -c ./configs/tiered/softmax/$model --tune-lmd $tune --meta-val-query $query --data ./data/tiered-imagenet/data --slk --knn 5 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-5.log --evaluate --enlarge $enlarge
