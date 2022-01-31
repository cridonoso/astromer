#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
data_name = sys.argv[2]
train_astromer = eval(sys.argv[3])
fold_n = int(sys.argv[4])

datasets = ['{}'.format(data_name)]

astroweights = './weights/astromer/finetuning'

target = './latest/training_astromer/' if train_astromer else './latest/freezed_astromer/'

for dataset in datasets:
    for mode in ['0','1','2']:
        print('{} on mode {}'.format(dataset, mode))
        command1 = 'python -m presentation.scripts.backprop \
                        --data ./data/records/{}/fold_{}/{} \
                        --w {}/{}_f{} \
                        --p {}/{}/fold_{}/{} \
                        --batch-size 512 \
                        --mode {} \
                        --gpu {}'.format(data_name, fold_n, dataset,
                                         astroweights, dataset, fold_n,
                                         target, data_name, fold_n, dataset,
                                         mode,
                                         gpu)
        if train_astromer:
            command1 += ' --finetune'

        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)
