#!/usr/bin/python
import sys
import subprocess
import time

root = '.presentation.scripts'
max_obs = 50
models = ['lstm_att', 'mlp_att', 'lstm']
astromer_weigths = '/tf/astromer/runs/machito_0/finetuning/model'
datapath = '/tf/astromer/data/records/ogle'
project_path = '/tf/astromer/clf_runs/machito/'

for mode, name in enumerate(models):
    start = time. time()
    command1 = 'python -m {}.classification --data {} \
                                               --max-obs {} \
                                               --w {} \
                                               --p {}{} \
                                               --mode {}'.format(root,
                                                                 datapath,
                                                                 max_obs, 
                                                                 astromer_weigths,
                                                                 project_path, name,
                                                                 mode)
    print('executing: ',command1)
    try:
        subprocess.call(command1, shell=True)
    except:
        print('ERROR IN: ',command1)
    end = time. time()
    print('{} takes {}'.format(name, (end - start)))