#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu        = sys.argv[1]
pt_folder  = sys.argv[2] #until finetuning
batch_size = 512    
records_folder = './data/records/'
ds_names = ['alcock', 'atlas']
spc_list = [20, 100]
clf_names = ['att_mlp', 'cls_mlp']

try:
    exp_name   = sys.argv[3] 
except:
    exp_name = 'classification'
    
root = 'python -m presentation.experiments.astromer_2.classify'
for dataset in ds_names:

    for clf_name in clf_names:
        for spc in spc_list:
            for fold_n in range(3):
                start = time.time()
                project_path = '{} --gpu {} --data {} --pt-folder {}  --ft-folder {} --clf-folder {} --clf-name {}'

                DATAPATH = os.path.join(records_folder,                                  
                                        dataset,
                                        'fold_'+str(fold_n), 
                                        '{}_{}'.format(dataset, spc))  

                FTWEIGTHS = os.path.join(pt_folder, 
                                         '..',
                                         'finetuning',                                    
                                         dataset,
                                         'fold_'+str(fold_n), 
                                         '{}_{}'.format(dataset, spc))   

                CLFWEIGHTS = os.path.join(pt_folder,
                                          'classification',                                     
                                          dataset,
                                          'fold_'+str(fold_n), 
                                          '{}_{}'.format(dataset, spc))

                command1 = project_path.format(root, gpu, DATAPATH, pt_folder, FTWEIGTHS, CLFWEIGHTS, clf_name)

                try:
                    subprocess.call(command1, shell=True)
                except Exception as e:
                    print(e)

                end = time. time()
                print('{} takes {:.2f} sec'.format(dataset, (end - start)))
