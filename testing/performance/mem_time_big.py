import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing
import numpy as np
import time
import os
import psutil
import tracemalloc
import os, sys
import sys
sys.path.append('/home/Samsung2TB/')
sys.path.append('/home/Samsung2TB/astromer/')
from astromer.src.data.record import DataPipeline
import os.path
import tensorflow as tf 

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=config)

METAPATH = './big_metadata_sample.parquet'
LCDIR = '../..//data/raw_data/macho_r_parquets/parquets/'
metadata = pd.read_parquet(METAPATH)
metadata['Class'] = pd.Categorical(metadata['Class'])
metadata['Label'] = metadata['Class'].cat.codes
metadata['Path'] = metadata['Path'].apply(lambda x: os.path.join(LCDIR, x))

file_exists = os.path.exists('time_memory_big.csv')
if file_exists is False:
    df = pd.DataFrame(columns = ['Time[s]', 'Memory[MB]'])
else: df = pd.read_csv('time_memory_big.csv', index_col=[0])

pipeline = DataPipeline(metadata=metadata, 
                        context_features=['ID', 'Label', 'Class'],
                        sequential_features=['mjd', 'mag'],)

st = time.time()
tracemalloc.start()

var = pipeline.run(LCDIR, METAPATH , n_jobs=int(sys.argv[1]))

current, peak = tracemalloc.get_traced_memory()
current = current/(1024*1024)
peak =  peak/(1024*1024)
et = time.time()
elapsed_time = et - st

tracemalloc.stop()
print('Execution time:', elapsed_time, 'seconds')
df.loc[len(df)] = [round(elapsed_time, 4)  , round(peak, 4) ]
#print(len(var))
del var
df.to_csv('time_memory_big.csv')
