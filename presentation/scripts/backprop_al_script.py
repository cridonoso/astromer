#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
data_name = sys.argv[2]
# fold_n = sys.argv[3]

datasets = ['{}'.format(data_name)]
astroweights = './weights/astromer_10022021/finetuning'
for dataset in datasets:
    for fold_n in [2,1,0]:
        for mode in ['2','1','0']:
            print('{} on mode {}'.format(dataset, mode))
            command1 = 'python -m presentation.scripts.backprop \
                            --data ./data/records/{}/fold_{}/{} \
                            --w {}/{}_f0 \
                            --p ./experiments_6/{}/fold_{}/{} \
                            --batch-size 512 \
                            --mode {} \
                            --gpu {}'.format(data_name, fold_n, dataset,
                                             astroweights, dataset,
                                             data_name, fold_n, dataset,
                                             mode,
                                             gpu)
            try:
                subprocess.call(command1, shell=True)
            except Exception as e:
                print(e)
