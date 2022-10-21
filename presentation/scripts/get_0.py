import tomli, tomli_w
import os

from datetime import datetime
# ==============================================================================
# GENERAL ======================================================================
# ==============================================================================
template_path  = './presentation/pipeline/config/pretraining.toml'
pretrained_weights = './weights/rf_atlas'
# ==============================================================================
# DATA INFO ====================================================================
# ==============================================================================
dataset_name    = 'atlas'
data_finetuning = './data/records/{}'.format(dataset_name)
data_classification = data_finetuning
# ==============================================================================
# FOLDERS ======================================================================
# ==============================================================================
save_weights_finetuning     = './results/finetuning/{}'.format(dataset_name)
save_weights_classification = './results/classification/{}'.format(dataset_name)
dir_to_save_config     = './presentation/pipeline/config/{}'.format(dataset_name)
os.makedirs(dir_to_save_config, exist_ok=True)
# ==============================================================================
# TRAINING HYPERPARAMETERS  ====================================================
# ==============================================================================
creation_date  = datetime.today().strftime('%Y-%m-%d')
batch_size_ft  = 2500
batch_size_clf = 512

# ==============================================================================
# =========================== SCRIPT BEGIN =====================================
# ==============================================================================
with open(template_path, mode="rb") as fp:
    config = tomli.load(fp)

for sci_case in ['a', 'b', 'c']:
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


            config['general']['creation_date'] = creation_date

            config['finetuning']['batch_size']     = batch_size_ft
            config['classification']['batch_size'] = batch_size_clf

            config['finetuning']['data']['path']     = ft_data
            config['classification']['data']['path'] = clf_data

            config['finetuning']['exp_path']     = ft_path
            config['classification']['exp_path'] = clf_path

            config['finetuning']['weights']     = pretrained_weights
            config['classification']['weights'] = ft_path # Classification uses finetuned weights

            final_folder_to_save_conf = os.path.join(dir_to_save_config, sci_case)
            os.makedirs(final_folder_to_save_conf, exist_ok=True)
            with open('{}/{}_{}.toml'.format(final_folder_to_save_conf,
                                             samples_per_class, fold_n), mode="wb") as fp:
                tomli_w.dump(config, fp)
