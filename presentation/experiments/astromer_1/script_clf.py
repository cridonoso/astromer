#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu        = sys.argv[1]
pt_folder  = sys.argv[2] #until pretraining
batch_size = 512    
records_folder = './data/records/'
ds_names = ['alcock', 'atlas']
spc_list = [20, 100]
clf_names = ['att_mlp']

try:
    exp_name   = sys.argv[3] 
except:
    exp_name = 'finetuning'
    
root = 'python -m presentation.experiments.astromer_1.classify'
for dataset in ds_names:
    print(dataset)
    for clf_name in clf_names:
        for spc in spc_list:
            for fold_n in range(3):
                start = time.time()
                project_path = '{} --gpu {} --subdataset {} --pt-folder {} --fold {} --spc {} --clf-name {} --ft-name {}'

                command1 = project_path.format(root, gpu, dataset, pt_folder, fold_n, spc, clf_name, exp_name)

                try:
                    subprocess.call(command1, shell=True)
                except Exception as e:
                    print(e)

                end = time. time()
                print('{} takes {:.2f} sec'.format(dataset, (end - start)))