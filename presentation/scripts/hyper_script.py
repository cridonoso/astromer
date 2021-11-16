#!/usr/bin/python
import subprocess
import sys
import time
import json
import os, sys

gpu = sys.argv[1]
data_name = sys.argv[2]

for mode in [ 'mlp_att']: 
    print('{} on mode {}'.format(data_name, mode))

    command1 = 'python -m presentation.scripts.hyper \
                    --data ./embeddings/{}/fold_0/{} \
                    --gpu {} \
                    --mode {} '.format(data_name, data_name,
                                    gpu,
                                    mode)
    try:
        subprocess.call(command1, shell=True)
    except Exception as e:
        print(e)
