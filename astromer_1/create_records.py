import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys,os

from src.data.record import create_dataset


from src.data import pretraining_pipeline 

def magtoflux(mag):
    return 10 ** (-0.4 * mag)



train_sets = list(range(40000, 49991, 1000))

for sets in train_sets:
    name                 = 'macho'
    lightcurves_folder   = f'./data/raw_data/{name}/LCs/' # lightcurves folder
    lightcurves_metadata = f'./data/raw_data/{name}/new_variability_R_snr.csv' # metadata file
    
    fold_to_save_records = './records/1609/filtered_macho_{}'.format(str(sets))
    meta = pd.read_csv(lightcurves_metadata)

    b = (meta.Std_r>0.1)
    metadata_filtered = meta[b] 
    metadata_filtered = metadata_filtered.sort_values(by='stdstd_p', ascending=False)
    metadata_filtered = metadata_filtered[0:sets]
    target = f'{fold_to_save_records}/{name}_pt_train'

    create_dataset(metadata_filtered, 
                lightcurves_folder, 
                target, 
                max_lcs_per_record=20000, 
                n_jobs=32,  
                subsets_frac=(1, 0))
    
    
    
    new_meta = meta[~meta['ID'].isin(metadata_filtered['ID'])]
    val_set = int(sets *  0.43)
        
    print('train_set', sets)
    print('val_set', val_set)

    new_meta = new_meta.sample(frac=1).reset_index(drop=True)
    meta_val = new_meta.sample(n=val_set)
    target = f'{fold_to_save_records}/fold_{str(fold)}/{name}_pt_val'

    create_dataset(meta_val, 
                    lightcurves_folder, 
                    target, 
                    max_lcs_per_record=20000, 
                    n_jobs=32, # depends on the number of cores 
                    subsets_frac=(0, 0.9))


