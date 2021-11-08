#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
data_name = sys.argv[2]
datasets = ['{}_20'.format(data_name), 
            '{}_50'.format(data_name), 
            '{}_100'.format(data_name), 
            '{}_500'.format(data_name), 
            '{}'.format(data_name)]

for dataset in datasets:
    for mode in ['0', '1', '2']: 
        print('{} on mode {}'.format(dataset, mode))
        if mode == '2':
            patience = 200
        else:
            patience = 50
            
        command1 = 'python -m presentation.scripts.classification \
                        --data ./embeddings/{} \
                        --p ./experiments/{} \
                        --mode {} \
                        --patience {} \
                        --gpu {}'.format(dataset, 
                                              dataset, 
                                              mode, 
                                              patience,
                                              gpu)
        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)
