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
#'{}'.format(data_name)
for dataset in datasets:
    for mode in ['0', '1', '2']: 
        print('{} on mode {}'.format(dataset, mode))
        if mode == '0':
            patience = 50
            conf = '{}_lstm_att.json'.format(data_name)
        if mode == '1':
            patience = 50
            conf = '{}_mlp_att.json'.format(data_name)
        if mode == '2':
            patience = 50
            conf = '{}_lstm.json'.format(data_name)
        
        command1 = 'python -m presentation.scripts.classification \
                        --data ./embeddings/{}/fold_{}/{} \
                        --p ./experiments/{}/fold_{}/{} \
                        --mode {} \
                        --patience {} \
                        --gpu {}'.format(data_name, fold_n, dataset,
                                         data_name, fold_n, dataset,
                                         mode, 
                                         patience,
                                         conf,
                                         gpu)
        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)

