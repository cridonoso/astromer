''' 

Testing the pretrained model in other datasets 

'''

from src.models.astromer_1 import get_ASTROMER, test_step, train_step
from src.data import load_data
from src.data.zero import pretraining_pipeline

import toml
import yaml
import os


def merge_metrics(**kwargs):
	merged = {}
	for key, value in kwargs.items():
		for subkey, subvalue in value.items():
			merged['{}_{}'.format(key, subkey)] = subvalue
	return merged


def average_logs(logs):
	N = len(logs)
	average_dict = {}
	for key in logs[0].keys():
		sum_log = sum(log[key] for log in logs)
		average_dict[key] = float(sum_log/N)
	return average_dict

###########################################################

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#ds_names = ['alcock', 'ogle', 'atlas']
#ds_names = ['kepler', 'kepler_alcock_linear', 'kepler_atlas_linear', 'kepler_ogle_linear']
ds_names = ['alcock', 'ogle', 'atlas', 'kepler', 'kepler_alcock_linear', 'kepler_atlas_linear', 'kepler_ogle_linear']
#ds_names = ['kepler']

folds = [0, 1, 2]
spc_list = [50]
#spc_list = ['all']

ROOT = './presentation/experiments/astromer_1_pe'
id_exp = 'exp_004_a/lr_scheduler'
ft_science_cases = ['PE'] #, 'FF1_PE', 'FF1_ATT_FF2', 'FF1_PE_ATT_FF2']
ft_science_cases = ['FF1_ATT_FF2', 'FF1_PE_ATT_FF2']

for ft_science_case in ft_science_cases:
    for spc in spc_list:
        for ds_name in ds_names:    
            for fold in folds:    
                print('Testing {} - fold {}'.format(ds_name.upper(), fold))

                if isinstance(spc, str):
                    path_data = './data/records/{}/fold_{}/{}'.format(ds_name, fold, ds_name)
                    data_name = ds_name
                else:
                    path_data = './data/records/{}/fold_{}/{}_{}'.format(ds_name, fold, ds_name, spc)
                    data_name = '{}_{}'.format(ds_name, spc)

                ft_folder = 'results/finetuning/P02R01_clean/{}/{}/{}/fold_{}/{}'.format(id_exp, 
                                                                                         ft_science_case,
                                                                                         ds_name,
                                                                                         fold,
                                                                                         data_name)

                test_step_fn = test_step

                with open('{}/{}/config.yaml'.format(ROOT, ft_folder)) as fp:
                    config = yaml.load(fp, Loader=yaml.FullLoader)

                with open('{}/{}/pe_config.yaml'.format(ROOT, ft_folder)) as fp:
                    pe_config = yaml.load(fp, Loader=yaml.FullLoader)


                astromer = get_ASTROMER(num_layers=config['Pretraining']['num_layers'],
                                        num_heads=config['Pretraining']['num_heads'],
                                        head_dim=config['Pretraining']['head_dim'],
                                        mixer_size=config['Pretraining']['mixer'],
                                        dropout=config['Pretraining']['dropout'],
                                        pe_type=config['Pretraining']['pe_type'],
                                        pe_config=pe_config,
                				        pe_func_name=config['Pretraining']['pe_func_name'],
				                        residual_type=config['Pretraining']['residual_type'],
                                        window_size=config['Pretraining']['window_size'],
                                        encoder_mode=config['Pretraining']['encoder_mode'],
                                        average_layers=config['Pretraining']['avg_layers'],
                                        data_name=None) # Funciona mejor cuando reescalo denuevo

                astromer.load_weights(os.path.join(ROOT, ft_folder, 'weights', 'weights'))

                test_loader = pretraining_pipeline(os.path.join(path_data, 'test'),
                                                   batch_size=5 if config['Pretraining']['debug'] else config['Pretraining']['bs'],
                                                   window_size=config['Pretraining']['window_size'],
                                                   shuffle=False,
                                                   sampling=False,
                                                   msk_frac=config['Pretraining']['probed'],
                                                   rnd_frac=config['Pretraining']['rs'],
                                                   same_frac=config['Pretraining']['rs'],
                                                   key_format='1')

                test_logs = []
                for x, y in test_loader:
                    logs = test_step_fn(astromer, x, y)
                    test_logs.append(logs)

                test_metrics = average_logs(test_logs)

                path_save = os.path.join(ROOT, ft_folder, 'test_metrics.toml')

                with open(os.path.join(path_save), 'w') as fp:
                    toml.dump(test_metrics, fp)

