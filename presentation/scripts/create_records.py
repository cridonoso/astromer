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

from src.data.record import DataPipeline

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
    metadata['Class'] = pd.Categorical(metadata['Class'])
    metadata['Label'] = metadata['Class'].cat.codes
    metadata['Path']  = metadata['Path'].apply(lambda x: os.path.join(OBSPATH, x)) 
    

    if opt.debug:
        metadata = metadata.sample(20)
        print('[INFO] Debugging: ', metadata.shape)
        
    custom_pipeline = CustomCleanPipeline(metadata=metadata,
                                          config_path=opt.config)

    if 'test' in metadata.sset.unique():
        test_metadata = metadata[metadata['sset'] == 'test']
        test_metadata = [test_metadata]*opt.folds
    else:
        test_metadata = None


    custom_pipeline.train_val_test(val_frac=opt.val_frac, 
                                   test_meta=test_metadata, 
                                   k_fold=opt.folds)

    var = custom_pipeline.run(observations_path=OBSPATH, 
                              metadata_path=METAPATH,
                              n_jobs=16,
                              elements_per_shard=opt.elements_per_shard)

    end = time.time()
    print('\n [INFO] ELAPSED: ', end - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/raw_data_parquet/alcock', type=str,
                    help='raw_data folder (parquet only)')
    parser.add_argument('--config', default='./data/raw_data_parquet/alcock/config.toml', type=str,
                    help='Config file specifying context and sequential features')

    parser.add_argument('--target', default='./data/records_parquet/', type=str,
                    help='target folder to save records files')


    parser.add_argument('--folds', default=1, type=int,
                    help='number of folds')
    parser.add_argument('--val-frac', default=0.2, type=float,
                    help='Validation fraction')
    parser.add_argument('--test-frac', default=0.2, type=float,
                    help='Validation fraction')

    parser.add_argument('--elements-per-shard', default=20000, type=int,
                    help='Number of light curves per shard')


    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


    opt = parser.parse_args()        
    run(opt)
