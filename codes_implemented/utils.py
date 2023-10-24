import glob

import tensorflow as tf 
import numpy as np
import pandas as pd

from src.data.record import process_lc3, deserialize

def open_all_records(path_subset):
    """Abre todos los records desde un subconjunto"""
    all_records = dict()
    
    path_objects = glob.glob('{}/*'.format(path_subset))

    for path_object in path_objects:
        path_records = glob.glob('{}/*'.format(path_object))

        cont = 0
        object_astro = path_object.split('/')[-1]
        print('Number of records within a class: {}'.format(object_astro))
        for path_record in path_records:
            print(f" - opening: {path_record}")
            dataset_train = tf.data.TFRecordDataset(path_record)
            dataset_train = dataset_train.map(deserialize)
            for lc in dataset_train:
                all_records[lc['lcid'].numpy()] = lc['input']              
                
            cont += 1
            
    return all_records


def open_subsets(path_subset_train, path_subset_val, path_subset_test):
    dataset_train = open_all_records(path_subset_train)
    dataset_val = open_all_records(path_subset_val)
    dataset_test = open_all_records(path_subset_test)
    
    return dataset_train, dataset_val, dataset_test


def distribution_values(subset):
    mag_lcs = []
    mjd_lcs_diff = []

    for _, lc in subset.items():
        mjd_lc = lc[:,0]
        mag_lc = lc[:,1]

        mjd_lcs_diff.append(np.diff(mjd_lc, axis=0))
        mag_lcs.append(mag_lc)
            
    return np.concatenate(np.array(mag_lcs)), np.concatenate(np.array(mjd_lcs_diff))


def normalization_mjd(subset):
    mjd_lcs = []
    for _, lc in subset.items():
        mjd_lc = lc[:,0]
        min_mjd = np.min(mjd_lc)
        mjd_lc = mjd_lc - min_mjd

        mjd_lcs.append(mjd_lc)

    return np.concatenate(np.array(mjd_lcs))


def create_df_dataset(subsets, name_col):
    df_mag = pd.DataFrame(np.concatenate(subsets),
                          columns=[name_col])
    return df_mag