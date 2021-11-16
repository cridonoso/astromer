#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

data_name = sys.argv[1]
datasets = ['{}_20'.format(data_name), 
            '{}_50'.format(data_name), 
            '{}_100'.format(data_name), 
            '{}_500'.format(data_name), 
            ]
#'{}'.format(data_name)
astroweights = './weights/astromer_10022021/finetuning'

take = 5
for dataset in datasets:
    for name in ['train', 'val', 'test']:  
        for fold_n in range(3):
            start = time.time()
            command1 = 'python -m presentation.scripts.get_att \
                       --data ./data/records/{}/fold_{}/{}/{} \
                       --batch-size 1000 \
                       --w {}/{}_f{} \
                       --p ./embeddings/{}/fold_{}/{}/{}.h5 '.format(data_name, fold_n, dataset, name,
                                                                     astroweights, dataset,fold_n,
                                                                     data_name, fold_n, dataset, name)
            try:
                subprocess.call(command1, shell=True)
            except Exception as e:
                print(e)

            end = time. time()
            print('{} takes {:.2f} sec'.format(dataset, (end - start)))
