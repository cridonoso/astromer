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

from src.data.record import DataPipeline, create_config_toml

import time
import polars as pl

class CustomCleanPipeline(DataPipeline):
    def lightcurve_step(self, inputs):
        """
        Preprocessing applied to each light curve separately
        """
        # First feature is time
        inputs = inputs.sort(self.sequential_features[0], descending=True) 
        p99 = inputs.quantile(0.99, 'nearest')
        p01 = inputs.quantile(0.01, 'nearest')
        inputs = inputs.filter(pl.col('mag') < p99['mag'])
        inputs = inputs.filter(pl.col('mag') > p01['mag'])
        return inputs

    def observations_step(self):
        """
        Preprocessing applied to all observations. Filter only
        """
        fn_0 = pl.col("errmag") < 1.  # Clean the data on the big lazy dataframe
        fn_1 = pl.col("errmag") > 0.

        return fn_0 & fn_1

def run(opt):
    
    start = time.time()

    opt.data = os.path.normpath(opt.data)


    METAPATH = os.path.join(opt.data, 'metadata.parquet')
    OBSPATH  = os.path.join(opt.data, 'light_curves')
    
    metadata = pd.read_parquet(METAPATH)
    metadata['Band'] = metadata['Band'].astype(str)
    metadata['ID'] = metadata['ID'].astype(str)
    
    test_metadata  = metadata.sample(frac=0.2)
    rest_metadata  = metadata[~metadata['ID'].isin(test_metadata['ID'])]

    validation_metadata = rest_metadata.sample(frac=0.15)
    train_metadata = rest_metadata[~rest_metadata['ID'].isin(validation_metadata['ID'])]
    for n_samples in [1e2, 1e3, 1e4, 1e5, 1e6]:
        print('[INFO] Creating dataset with {} samples'.format(n_samples))
        selected_train = train_metadata.sample(n=int(n_samples))

        ds_name = opt.data.split('/')[-1]

        target_path = os.path.join(opt.target, ds_name, '{}'.format(int(n_samples)))
        
        create_config_toml(parquet_id='newID',
                           target=target_path,
                           context_features=['ID', 'Label', 'Class', 'N', 'Band', 'shard'],
                           sequential_features=['mjd', 'mag', 'errmag'],
                           context_dtypes=['string', 'integer', 'string', 'integer', 'string', 'integer'],
                           sequential_dtypes=['float', 'float', 'float'])


        selected_train['subset_0']      = ['train']*selected_train.shape[0]
        validation_metadata['subset_0'] = ['validation']*validation_metadata.shape[0]
        test_metadata['subset_0']       = ['test']*test_metadata.shape[0]
        final_metadata = pd.concat([selected_train, validation_metadata, test_metadata])

        print(target_path)
        return 
        pipeline = CustomCleanPipeline(metadata=final_metadata,
                                       config_path=os.path.join(target_path, 'config.toml'))


        var = pipeline.run(observations_path=OBSPATH, 
                           n_jobs=8,
                           elements_per_shard=20000)

        end = time.time()
        print('\n [INFO] ELAPSED: ', end - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/raw_data_parquet/alcock', type=str,
                    help='raw_data folder (parquet only)')
    parser.add_argument('--config', default='./data/raw_data_parquet/alcock/config.toml', type=str,
                    help='Config file specifying context and sequential features')

    parser.add_argument('--target', default='./data/records/', type=str,
                    help='target folder to save records files')


    parser.add_argument('--elements-per-shard', default=20000, type=int,
                    help='Number of light curves per shard')


    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


    opt = parser.parse_args()        
    run(opt)
