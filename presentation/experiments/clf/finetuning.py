#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
ds_name = sys.argv[2]
astroweights = sys.argv[3]

project_path = './presentation/experiments/clf/{}'.format(ds_name)

datasets = ['{}_20'.format(ds_name),
            '{}_50'.format(ds_name),
            '{}_100'.format(ds_name),
            '{}_500'.format(ds_name)]

for dataset in datasets:
    print(dataset)
    for fold_n in range(3):
        start = time.time()
        command1 = 'python -m presentation.scripts.train \
                   --data ./data/records/{}/fold_{}/{} \
                   --w {} \
                   --batch-size 2048 \
                   --p {}/fold_{}/{} \
                   --gpu {}'.format(ds_name, fold_n, dataset,
                                    astroweights,
                                    project_path, fold_n, dataset,
                                    gpu)
        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)

        end = time. time()
        print('{} takes {:.2f} sec'.format(dataset, (end - start)))
