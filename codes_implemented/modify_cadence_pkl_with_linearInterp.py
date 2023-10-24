import random
import shutil
import time
import glob
import os

import joblib

import pandas as pd
import numpy as np

import scipy 

"""Cambiar el como se guardan los archivos, es decir, hacer que se guarden como datasets y no como records_cadence_modified"""

def modified_unlabeled_data_cadence(path_subsets, step_rm_rows, name_original_data, name_new_data, type_cadence):
    for path_subset in path_subsets:
        print(f"Subset: {path_subset}")
        path_subset_str = path_subset.split('/')
        if path_subset_str[-1] == 'objects.csv':
            path_save_csv = path_subset.replace('pickles', name_new_data)
            shutil.copy(path_subset, '/'.join(path_save_csv.split('/')[:-1]))
            continue
            
        path_objects = glob.glob('{}/*'.format(path_subset))
        
        for path_object in path_objects:
            path_records = glob.glob('{}/*'.format(path_object))

            for path_record in path_records:
                modify_record_cadence(path_record, step_rm_rows, type_cadence, name_original_data, name_new_data)

                
def modified_labeled_data_cadence(path_files_by_folds, step_rm_rows, name_original_data, name_new_data, type_cadence):
    for path_files_by_fold in path_files_by_folds:
        print(f"Fold: {path_files_by_fold}")
        path_shots = glob.glob('{}/*'.format(path_files_by_fold))

        for path_shot in path_shots:
            path_subsets = glob.glob('{}/*'.format(path_shot))
            
            for path_subset in path_subsets:
                path_subset_str = path_subset.split('/')

                if path_subset_str[-1] == 'objects.csv':
                    path_save_csv = path_subset.replace('records', name_new_data)
                    path_save_csv_aux = '/'.join(path_save_csv.split('/')[:-1])
                    os.makedirs(path_save_csv_aux, exist_ok=True)
                    shutil.copy(path_subset, path_save_csv_aux)
                    continue
                    
                path_objects = glob.glob('{}/*'.format(path_subset))
                
                for path_object in path_objects:
                    path_records = glob.glob('{}/*'.format(path_object))

                    for path_record in path_records:
                        modify_record_cadence(path_record, step_rm_rows, type_cadence, name_original_data, name_new_data)

                
def modify_record_cadence(path_record, step_rm_rows, type_cadence, name_original_data, name_new_data='alcock_2-4_random'):
    path_record = path_record.replace('\\', '/')
    path_save_record = path_record.replace(name_original_data, name_new_data)
    os.makedirs('/'.join(path_save_record.split('/')[:-1]), exist_ok=True)

    # Abre TFRecord
    dataset = pd.read_pickle(path_record)

    list_data_lcs = []
    for  i, lc_info in dataset.iterrows():
        #print(i)
        np_lc = lc_info['lc_data']
        np_lc = np_lc[np_lc[:,0].argsort()]
    
        # filtra cada cierta cantidad de pasos de manera random o controlada
        # Random
        if isinstance(step_rm_rows, str):
            if step_rm_rows == 'random':
                idx = range(0, np_lc.shape[0])
                fraction_cadence = type_cadence.split('/')
                idx_filter = sorted(random.sample(idx, np_lc.shape[0]*int(fraction_cadence[0])//int(fraction_cadence[1])))
                mask = np.ones(np_lc.shape[0], dtype=bool)
                mask[idx_filter] = False
                np_lc = np_lc[mask]
            
            elif step_rm_rows == 'kde':
                firts_mjd = np_lc[:,0][0]
                maximum = np.max(np_lc[:,0])

                if type_cadence == 'alcock':
                    min =  0.0
                    median = 1.930
                elif type_cadence == 'atlas': 
                    min = 0.0
                    median = 0.016
                elif type_cadence == 'ogle':
                    min =  0.001
                    median = 1.094

                random_value = random.uniform(0, 1)
                if random_value < 0.5:
                    t_value = np.random.choice(np.linspace(min, median, 50), 1)[0]
                else:
                    t_value = median

                try:
                    kde_load_model = joblib.load('./models_kde/model_{}_kde.joblib'.format(type_cadence))
                    obs_cadence = kde_load_model.sample(np_lc.shape[0]-1)
                    obs_cadence[obs_cadence < 0] = t_value   

                    func_interp = scipy.interpolate.interp1d(np_lc[:,0], np_lc[:,1], kind='linear', bounds_error=False, fill_value=0.)
                    func_interp_err = scipy.interpolate.interp1d(np_lc[:,0], np_lc[:,2], kind='linear', bounds_error=False, fill_value=0.)        
                    
                    mjd_modified = np.add.accumulate(obs_cadence.flatten()) + np.array([firts_mjd])
                    mjd_modified = mjd_modified[mjd_modified < maximum]
                    
                    np_lc_aux = np.zeros(shape=(mjd_modified.shape[0] + 1, np_lc.shape[1]))
                    np_lc_aux[:,0] = np.concatenate((np.array([firts_mjd]), mjd_modified))
                    np_lc_aux[:,1] = func_interp(np_lc_aux[:,0])
                    np_lc_aux[:,2] = func_interp_err(np_lc_aux[:,0])
                    np_lc = np.copy(np_lc_aux)

                # Si la curva solo tiene un punto entonces dejaremos ese mismo valor    
                except ValueError:
                    pass
                #func_interp = scipy.interpolate.CubicSpline(np_lc[:,0], np_lc[:,1])
                #func_interp_err = scipy.interpolate.CubicSpline(np_lc[:,0], np_lc[:,2])
                #print(np_lc)

        # Controlada
        else:
            if type_cadence == '3/4':
                idx_filter = np.arange(step_rm_rows, np_lc.shape[0], step_rm_rows+1)
                mask = np.ones(np_lc.shape[0], dtype=bool)
                mask[idx_filter] = False
                np_lc = np_lc[mask]

            else:
                np_lc = np_lc[0::step_rm_rows]

        # save .records modified
        data = {'lcid': lc_info['lcid'], 
                'lc_data': [np_lc],
                'label': lc_info['label']} 
        
        list_data_lcs.append(pd.DataFrame(data))

    df_dataset = pd.concat(list_data_lcs).reset_index(drop=True)
    df_dataset.to_pickle('{}'.format(path_save_record))
                     

                
def modified_cadence(name_original_data, step_rm_rows, name_new_data, type_cadence, path_initial='data'):    

    directories = [path_initial, 
                   'pickles/{}'.format(name_new_data)]
    
    path_concatenated = '.'
    for path in directories:
        path_concatenated += '/' + path
        os.makedirs('{}'.format(path_concatenated), exist_ok=True)

    path_files_by_folds = glob.glob('./{}/pickles/{}/*'.format(path_initial, name_original_data))

    print(path_files_by_folds)
    
    if name_original_data == 'macho':
        modified_unlabeled_data_cadence(path_files_by_folds, step_rm_rows, name_original_data, name_new_data, type_cadence)

    else: # ALCOCK, ATLAS, OGLE
        modified_labeled_data_cadence(path_files_by_folds, step_rm_rows, name_original_data, name_new_data, type_cadence)


if __name__ == '__main__':
    # Combinaciones:
    ## name_new_data='records_cadence_modified_1-4'
    ## step_rm_rows=3
    ## type_cadence='1/4'

    ## name_new_data='records_cadence_modified_2-4'
    ## step_rm_rows=2
    ## type_cadence='2/4'

    ## name_new_data='records_cadence_modified_3-4'
    ## step_rm_rows=3
    ## type_cadence='3/4'

    ## name_new_data='records_cadence_modified_random_2-4'
    ## step_rm_rows='random'
    ## type_cadence='2/4' 
    ## con '3/4' obtengo --> '1/4' (25% de las muestras)
    ## con '1/4' obtengo --> '3/4' (75% de las muestras)

    ## name_new_data='alcock_atlas'
    ## step_rm_rows='kde'
    ## type_cadence='atlas'

    ## name_new_data='alcock_ogle'
    ## step_rm_rows='kde'
    ## type_cadence='ogle'


    start = time.time()
    modified_cadence(name_original_data='alcock', step_rm_rows='kde', 
                     name_new_data='alcock_ogle_linear',
                     type_cadence='ogle')
    end = time.time()

    print(f"Tiempo de extracción {end-start}")