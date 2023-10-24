import random
import shutil
import time
import glob
import os

import tensorflow as tf
import pandas as pd

from src.data.record import deserialize, process_lc3
                
def extract_data_records(path_record, name_new_file_data, name_old_file_data):
    path_record = path_record.replace('\\', '/')
    path_save_pickle = path_record.replace(name_old_file_data, name_new_file_data)
    if name_old_file_data == 'records':
        path_save_pickle = path_save_pickle.replace('.record', '.pkl')
        os.makedirs('/'.join(path_save_pickle.split('/')[:-1]), exist_ok=True)

        # Abre TFRecord
        dataset = tf.data.TFRecordDataset(path_record)
        dataset = dataset.map(deserialize)

        list_data_lcs = []

        for lc_info in dataset:
            np_lc = lc_info['input'].numpy()
            np_lc = np_lc[np_lc[:,0].argsort()]

            data = {'lcid': lc_info['lcid'].numpy(), 
                    'lc_data': [np_lc],
                    'label': lc_info['label'].numpy()}  

            list_data_lcs.append(pd.DataFrame(data))

        df_dataset = pd.concat(list_data_lcs).reset_index()
        df_dataset.to_pickle('{}'.format(path_save_pickle))

    else:
        path_save_pickle = path_save_pickle.replace('.pkl', '.record')
        os.makedirs('/'.join(path_save_pickle.split('/')[:-1]), exist_ok=True)

        # Recorre y guarda un chunk.records completo
        with tf.io.TFRecordWriter(path_save_pickle) as writer:

            # Abre TFRecord
            dataset = pd.read_pickle(path_record)

            list_data_lcs = []
            for  i, lc_info in dataset.iterrows():
                np_lc = lc_info['lc_data']
                np_lc = np_lc[np_lc[:,0].argsort()]

                # save .records modified
                process_lc3(lc_info['lcid'], lc_info['label'], np_lc, writer)


def read_records(path_files_by_folds, name_new_file_data, name_old_file_data):    
    for path_files_by_fold in path_files_by_folds:
        print(f"Fold: {path_files_by_fold}")
        path_shots = glob.glob('{}/*'.format(path_files_by_fold))

        for path_shot in path_shots:
            path_subsets = glob.glob('{}/*'.format(path_shot))
            
            for path_subset in path_subsets:
                path_subset = path_subset.replace('\\', '/')
                path_subset_str = path_subset.split('/')

                if path_subset_str[-1] == 'objects.csv':
                    path_save_csv = path_subset.replace(name_old_file_data, name_new_file_data)
                    path_save_csv_aux = '/'.join(path_save_csv.split('/')[:-1])

                    os.makedirs(path_save_csv_aux, exist_ok=True)
                    shutil.copy(path_subset, path_save_csv_aux)
                    continue
                    
                path_objects = glob.glob('{}/*'.format(path_subset))
                
                for path_object in path_objects:
                    path_records = glob.glob('{}/*'.format(path_object))

                    for path_record in path_records:
                        extract_data_records(path_record, name_new_file_data, name_old_file_data)


if __name__ == '__main__':

    name_original_data = 'alcock_ogle_cubic'
    name_old_file_data = 'pickles'
    name_new_file_data = 'records'
    path_initial = 'data'

    directories = [path_initial, 
                   '{}/{}'.format(name_new_file_data, name_original_data)]

    path_concatenated = '.'
    for path in directories:
        path_concatenated += '/' + path
        os.makedirs('{}'.format(path_concatenated), exist_ok=True)

    path_files_by_folds = glob.glob('./{}/{}/{}/*'.format(path_initial, name_old_file_data, name_original_data))

    start = time.time()
    read_records(path_files_by_folds, name_new_file_data, name_old_file_data)
    end = time.time()

    print(f"Tiempo de extracción {end-start}")