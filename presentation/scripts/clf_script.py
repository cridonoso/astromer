#!/usr/bin/python
import subprocess
import sys
import time
import json
import os

modes = ['lstm', 'lstm_att', 'mlp_att']
datasets = ['alcock']
astroweights = './runs/huge'
root_exp = './clf_runs/'

conf_file = os.path.join(astroweights, 'conf.json')
with open(conf_file, 'r') as handle:
    conf = json.load(handle)

for dataset in datasets:
	for mode, name in enumerate(modes):
		start = time. time()
		command1 = 'python -m presentation.scripts.classification \
                   --data ./data/records/{} \
                   --max-obs {} \
				   --w  {} \
				   --mode {} \
                   --take 100 \
				   --p {}/{}/{}'.format(dataset,
                                        conf['max_obs'],
                                        astroweights,
    								    mode,
                                        root_exp, dataset, name)
		try:
		    subprocess.call(command1, shell=True)
		except Exception as e:
		    print(e)
		end = time. time()
		print('{} using {} takes {:.2f} sec'.format(dataset, name, (end - start)))
