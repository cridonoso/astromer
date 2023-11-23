#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu        = sys.argv[1]
pt_folder  = sys.argv[2] #BEFORE pretraining

batch_size = 2000    
ds_names = ['alcock', 'atlas']
spc_list = [20, 100]

try:
    exp_name   = sys.argv[3] 
except:
    exp_name = 'finetuning'
    
root = 'python -m presentation.experiments.astromer_2.save_embeddings'
for dataset in ds_names:

    for spc in spc_list:
        for fold_n in range(3):
            start = time.time()
            project_path = '{} --gpu {} --data {} --pt-folder {} --target {} --bs 2000'
            
            exp_folder = pt_folder.split('/')[-2]
            print(exp_folder)
            DATADIR = './data/records/{}/fold_{}/{}_{}'.format(dataset, fold_n, dataset, spc)
            FTWEIGHTS = '{}/finetuning/{}/fold_{}/{}_{}'.format(pt_folder, dataset, fold_n, dataset, spc)
            TARGET = './data/embeddings/{}/{}/fold_{}/{}_{}'.format(exp_folder, dataset, fold_n, dataset, spc)

            command1 = project_path.format(root, gpu, DATADIR, FTWEIGHTS, TARGET)
            print(command1)
            try:
                subprocess.call(command1, shell=True)
            except Exception as e:
                print(e)

            end = time. time()
            print('{} takes {:.2f} sec'.format(dataset, (end - start)))