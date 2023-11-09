#!/usr/bin/python
import subprocess
import shutil
import time
import glob
import json
import sys
import os


gpu        = sys.argv[1]
pt_folder  = sys.argv[2] #until pretraining
batch_size = 2500    
records_folder = './data/records/'
ds_names = ['alcock', 'atlas']
spc_list = [20, 100]
clf_names = ['att_mlp', 'cls_mlp', 'all_mlp']

try:
    exp_name   = sys.argv[3] 
except:
    exp_name = 'finetuning'
    
root = 'python -m presentation.experiments.astromer_2.classify'
ckpt_weights = glob.glob(os.path.join(pt_folder, 'ckpts', '*'))

src = os.path.join(pt_folder, 'config.toml')

for ckpt_w in ckpt_weights:
    dst = os.path.join(ckpt_w, 'config.toml')
    shutil.copyfile(src, dst)
    for dataset in ds_names:
        print(dataset)
        for clf_name in clf_names:
            for spc in spc_list:
                for fold_n in range(3):
                    start = time.time()
                    project_path = '{} --gpu {} --subdataset {} --pt-folder {} --fold {} --spc {} --clf-name {} --ft-name {} --target-dir {}'
                    FTWEIGTHS = os.path.join(ckpt_w,
                                             exp_name, #'finetuning',                                     
                                             dataset,
                                             'fold_'+str(fold_n), 
                                             '{}_{}'.format(dataset, spc))   
                    command1 = project_path.format(root, gpu, dataset, ckpt_w, fold_n, spc, clf_name, exp_name, FTWEIGTHS)

                    try:
                        subprocess.call(command1, shell=True)
                    except Exception as e:
                        print(e)

                    end = time. time()
                    print('{} takes {:.2f} sec'.format(dataset, (end - start)))
