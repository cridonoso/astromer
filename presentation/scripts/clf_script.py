#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
data_name = sys.argv[2]
# fold_n = sys.argv[3]

datasets = ['{}_20'.format(data_name),
            '{}_50'.format(data_name),
            '{}_100'.format(data_name),
            '{}_500'.format(data_name),
            ]
patience = 100
# datasets = ['{}'.format(data_name)]
for dataset in datasets:
    for fold_n in range(3):
        for mode in ['0', '1', '2']: 
            print('{} on mode {}'.format(dataset, mode))
            command1 = 'python -m presentation.scripts.classification \
                            --data ./embeddings/{}/fold_{}/{} \
                            --p ./experiments_2/{}/{}/fold_{} \
                            --batch-size 256 \
                            --mode {} \
                            --patience {} \
                            --gpu {}'.format(data_name, fold_n, dataset,
                                             data_name, dataset, fold_n,
                                             mode,
                                             patience,
                                             gpu)
            try:
                subprocess.call(command1, shell=True)
            except Exception as e:
                print(e)
