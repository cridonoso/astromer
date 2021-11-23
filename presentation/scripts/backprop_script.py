#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
data_name = sys.argv[2]
fold_n = sys.argv[3]

datasets = ['{}_20'.format(data_name),
            '{}_50'.format(data_name),
            '{}_100'.format(data_name),
            '{}_500'.format(data_name),
            ]
# datasets = ['{}'.format(data_name)]
for dataset in datasets:
    for mode in ['0', '1', '2']:
        print('{} on mode {}'.format(dataset, mode))
        command1 = 'python -m presentation.scripts.backprop \
                        --data ./data/records/{}/fold_{}/{} \
                        --p ./experiments_3/{}/fold_{}/{} \
                        --batch-size 512 \
                        --mode {} \
                        --gpu {}'.format(data_name, fold_n, dataset,
                                         data_name, dataset, fold_n,
                                         mode,
                                         gpu)
        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)
