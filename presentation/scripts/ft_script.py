#!/usr/bin/python
import subprocess
import sys
import time
import json
import os

datasets = ['ogle_10',  'ogle_100',  'ogle_1000',  'ogle_2500',  'ogle_500']
astroweights = './runs/machito_0'

conf_file = os.path.join(astroweights, 'conf.json')
with open(conf_file, 'r') as handle:
    conf = json.load(handle)

for dataset in datasets:
    start = time.time()
    command1 = 'python -m presentation.scripts.finetuning \
               --data ./data/records/{} \
               --p {}\
    		   --prefix {}'.format(dataset,
                                   astroweights,
    							   dataset)
    try:
        subprocess.call(command1, shell=True)
    except Exception as e:
        print(e)
    end = time. time()
    print('{} takes {:.2f} sec'.format(dataset, (end - start)))
