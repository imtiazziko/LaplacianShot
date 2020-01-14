#!/bin/bash
#cd ./src
#python download_models.py
#python ./src/train.py -c ./configs/mini/softmax/resnet18.config --data ./data/images
#python ./src/train_slk.py -c ./configs/mini/softmax/resnet18.config --data ./data/images --slk --lmd 0.8 --knn 3 --log-file '/slk-knn-3-lambda-0.8-2-test.log'
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# train on mini imagenet
enlarge=True
# conv
model=conv4.config
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-1.0.log --evaluate --enlarge $enlarge

# Resnet 10
model=resnet10.config
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-1.0.log --evaluate --enlarge $enlarge

# Resnet 18
model=resnet18.config
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-1.0.log --evaluate --enlarge $enlarge

# WRN
model=wideres.config
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-1.0.log --evaluate --enlarge $enlarge

# Mobilenet
model=mobilenet.config
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-1.0.log --evaluate --enlarge $enlarge

# Densenet
model=densenet121.config
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.1 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.1.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.3 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.3.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.5 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.5.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.7 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.7.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 0.8 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-0.8.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 5 --log-file /enlarge-$enlarge/slk-knn-5-lambda-1.0.log --evaluate --enlarge $enlarge
python ./src/train_slk.py -c ./configs/mini/softmax/$model --data ./data/images --slk --lmd 1.0 --knn 3 --log-file /enlarge-$enlarge/slk-knn-3-lambda-1.0.log --evaluate --enlarge $enlarge