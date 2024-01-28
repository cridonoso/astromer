#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu        = sys.argv[1]
pt_folder  = sys.argv[2] #until pretraining
# all_visible = sys.argv[3].lower() == 'true'
try:
    exp_name   = sys.argv[3] 
except:
    exp_name = 'finetuning'
    

batch_size = 2500    
records_folder = './data/records/'
ds_names = ['alcock', 'atlas']
spc_list = [20, 100, 500]
 
root = 'python -m presentation.experiments.astromer_2.finetune'

for mode_ft in [' --allvisible', '']:
    if mode_ft == ' --allvisible':
        exp_name = exp_name+'_AV'

    for dataset in ds_names:
        print(dataset)
        for spc in spc_list:
            for fold_n in range(3):
                start = time.time()
                project_path = '{} --gpu {} --subdataset {} --pt-folder {} --fold {} --spc {} --exp-name {} --target-dir {}'
                FTWEIGTHS = os.path.join(pt_folder, 
                                         '..', 
                                         exp_name, 
                                         dataset, 
                                         'fold_'+str(fold_n), 
                                         '{}_{}'.format(dataset, spc))   
                command1 = project_path.format(root, gpu, dataset, pt_folder, fold_n, spc, exp_name, FTWEIGTHS)
                command1+=mode_ft
                    
                try:
                    subprocess.call(command1, shell=True)
                except Exception as e:
                    print(e)

                end = time. time()
                print('{} takes {:.2f} sec'.format(dataset, (end - start)))