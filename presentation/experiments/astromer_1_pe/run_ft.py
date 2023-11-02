#!/usr/bin/python
import pandas as pd
import numpy as np

import subprocess
import time
import sys, os

gpu = '1'
#ds_names = ['alcock', 'atlas', 'ogle']
#ds_names = ['kepler', 'kepler_alcock_linear', 'kepler_atlas_linear', 'kepler_ogle_linear']
#ds_names = ['alcock', 'atlas', 'ogle', 'kepler', 'kepler_alcock_linear', 'kepler_atlas_linear', 'kepler_ogle_linear']
ds_names = ['kepler']

folds = [0, 1, 2]
#spc_list = [500]
#spc_list = ['all']
spc_list = [50]

id_exp = 'pretrained_weights' # weights_pretrained
pt_folder = 'results/pretraining/P02R01/{}'.format(id_exp)

lr = 'scheduler'
bs = 1000
patience = 5
num_epochs = 10000

ft_folder = 'results/finetuning/P02R01/{}/lr_{}'.format(id_exp, lr)
#ft_science_cases = ['PE']
ft_science_cases = ['PE'] # , 'FF1_PE', 'FF1_ATT_FF2', 'FF1_PE_ATT_FF2']  # 'nontrain'
scale_pe_freq = False
debug = False

if scale_pe_freq:
    ft_folder += ft_folder + '_pe_by_mean'

ROOT = './presentation/experiments/astromer_1_pe'
root = 'python -m presentation.experiments.astromer_1_pe.scripts.finetuning'
for ft_science_case in ft_science_cases:
    for dataset in ds_names:
        print(dataset)
        for spc in spc_list:
            prom_time = []
            for fold in folds:
                start = time.time()
    
                command1 = '{} \
                            --gpu {} \
                            --dataset {} \
                            --fold {} \
                            --spc {} \
                            --pt-folder {} \
                            --ft-folder {} \
                            --ft-science-case {} \
                            --lr {} \
                            --bs {} \
                            --patience {} \
                            --num-epochs {}'.format(root,
                                                    gpu, 
                                                    dataset, 
                                                    fold,
                                                    spc,
                                                    pt_folder,
                                                    ft_folder, 
                                                    ft_science_case,
                                                    lr,
                                                    bs,
                                                    patience,
                                                    num_epochs
                                                    )

                if scale_pe_freq:
                    command1 += ' --scale-pe-freq'

                try:
                    subprocess.call(command1, shell=True)
                except Exception as e:
                    print(e)

                end = time.time()      
                prom_time.append(end - start)

                print('{} takes {:.2f} sec'.format(dataset, (end - start)))

            # Guarda el tiempo promedio de los 3 folds
            d = {'Mean time': [np.array(prom_time).mean()],
                 'Std dev': [np.array(prom_time).std()]}

            df_time = pd.DataFrame(data=d, index=[dataset])
            path_time = os.path.join(ROOT, ft_folder, ft_science_case, dataset)
            if isinstance(spc, str):
                df_time.to_csv('{}/time_{}.csv'.format(path_time, dataset), index=False)
            else:
                df_time.to_csv('{}/time_{}_{}.csv'.format(path_time, dataset, spc), index=False)



