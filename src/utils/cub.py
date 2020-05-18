import os
import numpy as np
import argparse
from os import listdir
from os.path import isfile, isdir, join
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to the data')
    parser.add_argument('--split', type=str, help='path to the split folder')
    args = parser.parse_args()
    dataset_list = ['train','val','test']
    #
    prex1 = args.data
    data_path = join(prex1,'images/')
#
    folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    folder_list.sort()
    label_dict = dict(zip(folder_list,range(0,len(folder_list))))

    classfile_list_all = []

    for i, folder in enumerate(folder_list):
        folder_path = join(data_path, folder)
        classfile_list_all.append( [join(folder,cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
        random.shuffle(classfile_list_all[i])

    if not os.path.isdir(args.split):
        os.makedirs(args.split)

breakpoint()
for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'train' in dataset:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

        if 'val' in dataset:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

        if 'test' in dataset:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    if 'train' in dataset:
        with open(args.split + '/train.csv', 'w') as f1:
            f1.writelines(['{},{}\n'.format(name[0], name[1]) for name in zip(file_list,label_list)])
    if 'val' in dataset:
        with open(args.split + '/val.csv', 'w') as f2:
            f2.writelines(['{},{}\n'.format(name[0], name[1]) for name in zip(file_list,label_list)])
    if 'test' in dataset:
        with open(args.split + '/test.csv', 'w') as f3:
            f3.writelines(['{},{}\n'.format(name[0], name[1]) for name in zip(file_list,label_list)])
