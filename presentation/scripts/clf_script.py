#!/usr/bin/python
import subprocess
import sys
import time
import json
import os

modes = ['lstm_att', 'mlp_att', 'lstm']
datasets = ['ogle_10',  'ogle_100',  'ogle_1000',  'ogle_2500',  'ogle_500']
astroweights = './runs/machito_0/finetuning'
root_exp = './clf_runs/'


for dataset in datasets:
    for mode, name in enumerate(modes):
        path_w = os.path.join(astroweights, dataset)

        conf_file = os.path.join(path_w, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)

        print(path_w)
        start = time.time()
        command1 = 'python -m presentation.scripts.classification \
                   --data ./data/records/{} \
                   --max-obs {} \
        		   --w  {} \
        		   --mode {} \
                   --take 100 \
        		   --p {}/{}/{}'.format(dataset,
                                        conf['max_obs'],
                                        path_w,
        							    mode,
                                        root_exp, dataset, name)
        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)
        end = time.time()
        print('{} using {} takes {:.2f} sec'.format(dataset, name, (end - start)))
