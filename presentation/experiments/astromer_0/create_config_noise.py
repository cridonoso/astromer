import tomli, tomli_w
import os

from datetime import datetime

# ==============================================================================
# OPENING TEMPLATE CONFIG FILE =================================================
# Template should contain all the parameters needed for running the pipeline
# (i.e., astromer hyperparameters)
# Parameters that change based on the datasets will be overwritten by this script
# ==============================================================================
template_path = './presentation/experiments/astromer_0/config_files/macho_mask.toml'
with open(template_path, mode="rb") as fp:
    config = tomli.load(fp)
# ==============================================================================
# GENERAL CONFIGURATION ========================================================
# ==============================================================================    
config['pretraining']['lr'] = 1e-5
config['pretraining']['scheduler']= False
name_opt = 'scheduler' if config['pretraining']['scheduler'] else 'LR{}'.format(config['pretraining']['lr'])

for nse in [0., 0.2, 0.4, 0.8, 1.]:
    config['masking']['mask_frac'] = 0.
    config['masking']['rnd_frac']  = nse
    config['masking']['same_frac'] = 0.
    config['positional']['alpha'] = 1
    norm_method = 'zero-mean'
    # ==============================================================================
    master_path         = './presentation/experiments/astromer_0' # shouldn't change
    master_name         = 'alcock_{}_r{}_{}_{}'.format(int(config['masking']['mask_frac']*100), 
                                                       int(config['masking']['rnd_frac']*100),
                                                          norm_method,
                                                          name_opt) # master name
    pretraining_data    = './data/records/alcock/fold_0/alcock' # unlabeled dataset
    dir_to_save_config  = f'{master_path}/config_files_noisy_alcock/{master_name}'
    dir_to_save_results = 'results_noisy_alcock'
    # ==============================================================================
    config['pretraining']['exp_path'] = f'{master_path}/{dir_to_save_results}/{master_name}/pretraining'
    config['pretraining']['data']['path'] = pretraining_data
    pretrained_weights  = config['pretraining']['exp_path'] # Change if pretrained weights already exists
    # ==============================================================================
    datasets_to_finetune = ['alcock', 'ogle', 'atlas']
    science_cases        = ['a']
    # ==============================================================================
    creation_date  = datetime.today().strftime('%Y-%m-%d')
    batch_size_ft  = 2500
    batch_size_clf = 512
    # ==============================================================================
    # CREATE CONFIG FILES ==========================================================
    # ==============================================================================
    os.makedirs(dir_to_save_config, exist_ok=True)

    config['pretraining']['data']['target'] = ''
    config['pretraining']['data']['fold'] = 0
    config['pretraining']['data']['spc'] = ''    
    config['pretraining']['data']['normalize'] = norm_method 

    for dataset_name in datasets_to_finetune:
        data_finetuning = f'./data/records/{dataset_name}'
        data_classification = data_finetuning

        save_weights_finetuning     = f'{master_path}/{dir_to_save_results}/{master_name}/{dataset_name}/finetuning/'
        save_weights_classification = f'{master_path}/{dir_to_save_results}/{master_name}/{dataset_name}/classification/'

        for sci_case in science_cases:
            if sci_case == 'a':
                config['classification']['train_astromer'] = False
            else:
                config['classification']['train_astromer'] = True

            for fold_n in range(3):
                for samples_per_class in [20, 50, 100, 500]:
                    ft_data  = os.path.join(data_finetuning,
                                            'fold_{}'.format(fold_n),
                                            '{}_{}'.format(dataset_name, samples_per_class))
                    clf_data = os.path.join(data_classification,
                                            'fold_{}'.format(fold_n),
                                            '{}_{}'.format(dataset_name, samples_per_class))

                    ft_path  = os.path.join(save_weights_finetuning,
                                            '{}_{}_f{}'.format(dataset_name, samples_per_class,fold_n))
                    clf_path = os.path.join(save_weights_classification,
                                            sci_case,
                                            '{}_{}_f{}'.format(dataset_name, samples_per_class,fold_n))
                    

                    config['finetuning']['data']['target'] = dataset_name
                    config['finetuning']['data']['fold'] = fold_n
                    config['finetuning']['data']['spc'] = samples_per_class
                    config['finetuning']['data']['normalize'] = norm_method 
                    config['classification']['data']['target'] = dataset_name
                    config['classification']['data']['fold'] = fold_n
                    config['classification']['data']['spc'] = samples_per_class
                    config['classification']['data']['normalize'] = norm_method 
                    
                    config['general']['creation_date'] = creation_date

                    config['finetuning']['batch_size']     = batch_size_ft
                    config['classification']['batch_size'] = batch_size_clf

                    config['finetuning']['data']['path']     = ft_data
                    config['classification']['data']['path'] = clf_data

                    config['finetuning']['exp_path']     = ft_path
                    config['classification']['exp_path'] = clf_path

                    config['finetuning']['weights']     = pretrained_weights
                    config['classification']['weights'] = ft_path # Classification uses finetuned weights

                    with open(f'{dir_to_save_config}/{dataset_name}.{samples_per_class}.f{fold_n}.{sci_case}.toml',
                              mode="wb") as fp:
                        tomli_w.dump(config, fp)
