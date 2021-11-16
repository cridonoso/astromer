#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
ds_name = sys.argv[2]
print(ds_name)
# datasets = ['{}_20'.format(ds_name), 
#             '{}_50'.format(ds_name), 
#             '{}_100'.format(ds_name), 
#             '{}_500'.format(ds_name), 
#             ]
datasets = ['{}'.format(ds_name)]

astroweights = './weights/astromer_10022021'

conf_file = os.path.join(astroweights, 'conf.json')
with open(conf_file, 'r') as handle:
    conf = json.load(handle)

for dataset in datasets:
    print(dataset)
    for fold_n in range(3):
        start = time.time()
        command1 = 'python -m presentation.scripts.finetuning \
                   --data ./data/records/{}/fold_{}/{} \
                   --p {} \
                   --prefix {}_f{} \
                   --gpu {}'.format(ds_name, fold_n, dataset,
                                    astroweights,
                                    dataset,fold_n,
                                    gpu)
        try:
            subprocess.call(command1, shell=True)
        except Exception as e:
            print(e)

        end = time. time()
        print('{} takes {:.2f} sec'.format(dataset, (end - start)))
