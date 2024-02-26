'''
This script was made exclusively to transform old data format to a new parquet-based one.
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import toml
import sys
import os


def run(opt):
    
    opt.data = os.path.normpath(opt.data)

    metadata = pd.read_csv(os.path.join(opt.data, 'train_metadata.csv'))
    metadata['sset'] = ['train']*metadata.shape[0]

    if os.path.exists(os.path.join(opt.data, 'test_metadata.csv')):
        test_metadata = pd.read_csv(os.path.join(opt.data, 'test_metadata.csv'))
        test_metadata['sset'] = ['test']*test_metadata.shape[0]
        metadata = pd.concat([metadata, test_metadata])

    if opt.debug:
        metadata = metadata.sample(20)

    dataset_name = os.path.basename(opt.data)
    target_dir = os.path.join(opt.target, dataset_name)
    light_curves_dir = os.path.join(target_dir, 'light_curves')
    os.makedirs(light_curves_dir, exist_ok=True)

    path = (opt.data + '/LCs/'+ metadata.Path).to_list()
    IDs = metadata.ID.to_list()
    metadata = metadata.assign(newID=metadata.index.values)

    dfs = []
    for cont, (file, id_) in enumerate(zip(path, IDs)):
        # Read the file
        df = pd.read_csv(file, engine='c', na_filter=False)
        df.columns = ['mjd', 'mag', 'errmag']
        df['newID'] = metadata.newID.iloc[cont]*np.ones(df.shape[0]).astype(np.int64)
        dfs.append(df)
    dfs = pd.concat(dfs) 
    dfs = dfs.set_index('newID')


    for batch, begin in enumerate(np.arange(0, dfs.shape[0], opt.samples_per_chunk)):
        df_sel = dfs.iloc[begin:begin+opt.samples_per_chunk]    
        n = str(batch).rjust(3, '0')
        df_sel.to_parquet(os.path.join(light_curves_dir, 'shard_'+n+'.parquet'))

    metadata['Label'] = pd.Categorical(metadata['Class']).codes
    metadata.to_parquet(os.path.join(target_dir, 'metadata.parquet'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/raw_data/alcock', type=str,
                    help='raw_data folder')
    parser.add_argument('--target', default='./data/raw_data_parquet/', type=str,
                    help='target folder to save parquet files')

    parser.add_argument('--samples-per-chunk', default=1000, type=int,
                    help='Number of light curves per chunk')

    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


    opt = parser.parse_args()        
    run(opt)
