import tomli, tomli_w
import os

from datetime import datetime

# ==============================================================================
# OPENING TEMPLATE CONFIG FILE =================================================
# Template should contain all the parameters needed for running the pipeline
# (i.e., astromer hyperparameters)
# Parameters that change based on the datasets will be overwritten by this script
# ==============================================================================
template_path       = './src/pipeline/template.toml'
with open(template_path, mode="rb") as fp:
    config = tomli.load(fp)
# ==============================================================================
# GENERAL CONFIGURATION ========================================================
# ==============================================================================
master_path         = './presentation/experiments/astromer_2' # shouldn't change
pe_c                = 1.
pretraining_data    = f'./data/records/macho' # unlabeled dataset
master_name         = f'macho_{int(pe_c)}' # master name
dir_to_save_config  = f'{master_path}/config_files/{master_name}'
# ==============================================================================
config['pretraining']['exp_path'] = f'{master_path}/results/{master_name}/pretraining'
config['pretraining']['data']['path'] = pretraining_data
pretrained_weights  = config['pretraining']['exp_path'] # Change if pretrained weights already exists
# ==============================================================================
datasets_to_finetune = ['alcock', 'ogle', 'atlas']
science_cases        = ['a']
subsets_to_train     = [20, 100, 500]
# ==============================================================================
creation_date  = datetime.today().strftime('%Y-%m-%d')
batch_size_ft  = 2500
batch_size_clf = 512
# ==============================================================================
# CREATE CONFIG FILES ==========================================================
# ==============================================================================
os.makedirs(dir_to_save_config, exist_ok=True)

config['astromer']['pe_c'] = pe_c
for dataset_name in datasets_to_finetune:
    data_finetuning = f'./data/records/{dataset_name}'
    data_classification = data_finetuning

    save_weights_finetuning     = f'{master_path}/results/{master_name}/{dataset_name}/finetuning/'
    save_weights_classification = f'{master_path}/results/{master_name}/{dataset_name}/classification/'

    for sci_case in science_cases:
        if sci_case == 'a':
            config['classification']['train_astromer'] = False
        else:
            config['classification']['train_astromer'] = True

        for fold_n in range(3):
            for samples_per_class in subsets_to_train:
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
