#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
ds_name = sys.argv[2]
science_case = sys.argv[3]
astromer_dim = 256
batch_size = 512

if science_case == 'a':
    train_astromer = False
    science_cases = ['a', 'b']
    sc_ft = 'ab'

if science_case == 'c':
    train_astromer = True
    science_cases = ['c']
    sc_ft = 'c'

datasets = ['{}_20'.format(ds_name),
            '{}_50'.format(ds_name),
            '{}_100'.format(ds_name),
            '{}_500'.format(ds_name),
            ]

for science_case in science_cases:
    for dataset in datasets:
        for fold_n in range(3):
            for mode in ['lstm_att', 'mlp_att', 'lstm']:
                print('sc:{} - {} on mode {}'.format(science_case, dataset, mode))
                astroweights = './runs/astromer_{}/{}/{}/fold_{}/{}'.format(astromer_dim,
                                                                            sc_ft,
                                                                            ds_name,
                                                                            fold_n,
                                                                            dataset)

                project_path = './runs/astromer_{}/classifiers/{}/{}/fold_{}/{}'.format(astromer_dim,
                                                                                        science_case,
                                                                                        ds_name,
                                                                                        fold_n,
                                                                                        dataset)

                command1 = 'python -m presentation.scripts.classify \
                                --data ./data/records/{}/fold_{}/{} \
                                --p {} \
                                --w {} \
                                --batch-size {} \
                                --mode {} \
                                --gpu {}'.format(ds_name, fold_n, dataset,
                                                 project_path,
                                                 astroweights,
                                                 batch_size,
                                                 mode,
                                                 gpu)
                if train_astromer:
                    command1 += ' --finetune'

                try:
                    subprocess.call(command1, shell=True)
                except Exception as e:
                    print(e)
