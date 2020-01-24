#!/bin/bash
#cd ./src
#python download_models.py
#python ./src/train.py -c ./configs/mini/softmax/resnet18.config --data ./data/images
#python ./src/train_slk.py -c ./configs/mini/softmax/resnet18.config --data ./data/images --slk --lmd 0.8 --knn 3 --log-file '/slk-knn-3-lambda-0.8-2-test.log'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
# train on mini imagenet
enlarge=True
query=15
tune=False
# conv
model=conv4.config
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.1 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.3 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.7 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.8 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.0 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.2 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.2.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.5.log --evaluate --enlarge $enlarge
# ##
# ## Resnet 10
model=resnet10.config
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.1 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.3 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.7 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.8 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.0 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.2 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.2.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.5.log --evaluate --enlarge $enlarge

# ## Resnet 18
model=resnet18.config
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.1 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.3 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.7 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.8 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.0 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.2 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.2.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.5.log --evaluate --enlarge $enlarge

### WRN
model=wideres.config
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.1 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.3 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.7 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.8 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.0 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.2 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.2.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.5.log --evaluate --enlarge $enlarge

# ### Mobilenet
model=mobilenet.config
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.1 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.3 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.7 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.8 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.0 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.2 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.2.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.5.log --evaluate --enlarge $enlarge

# ### Densenet
model=densenet121.config
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.1 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.3 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.7 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 0.8 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.0 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.2 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.2.log --evaluate --enlarge $enlarge
python ./src/train_slk_validation.py -c ./configs/mini/softmax/$model --lmd 1.5 --tune-lmd $tune --meta-val-query $query --data ./data/images --slk --knn 3 --log-file /enlarge-$enlarge-query-$query-validated/slk-knn-3-lmd-1.5.log --evaluate --enlarge $enlarge
