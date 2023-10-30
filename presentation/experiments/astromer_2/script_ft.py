#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu        = sys.argv[1]
pt_folder  = sys.argv[2] #until pretraining
all_visible = sys.argv[3].lower() == 'true'
try:
    exp_name   = sys.argv[4] 
except:
    exp_name = 'finetuning'
    

batch_size = 2500    
records_folder = './data/records/'
ds_names = ['alcock', 'atlas']
spc_list = [20, 100]
 
root = 'python -m presentation.experiments.astromer_2.finetune'
for dataset in ds_names:
    print(dataset)
    for spc in spc_list:
        for fold_n in range(3):
            start = time.time()
            
            project_path = '{} --gpu {} --subdataset {} --pt-folder {} --fold {} --spc {} --exp-name {}'
            command1 = project_path.format(root, gpu, dataset, pt_folder, fold_n, spc, exp_name)
            if all_visible:
                command1+=' --allvisible'
                
            try:
                subprocess.call(command1, shell=True)
            except Exception as e:
                print(e)

            end = time. time()
            print('{} takes {:.2f} sec'.format(dataset, (end - start)))