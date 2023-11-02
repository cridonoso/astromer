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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#ds_names = ['alcock', 'ogle', 'atlas']
#ds_names = ['kepler', 'kepler_alcock_linear', 'kepler_atlas_linear', 'kepler_ogle_linear']
ds_names = ['alcock', 'ogle', 'atlas', 'kepler', 'kepler_alcock_linear', 'kepler_atlas_linear', 'kepler_ogle_linear']
#ds_names = ['kepler']

folds = [0, 1, 2]
spc_list = [50]
#spc_list = ['all']
scale_pe_freq = False

ROOT = './presentation/experiments/astromer_1_pe'
pt_folder = 'results/pretraining/P02R01/pretrained_weights'

train_step_fn = train_step
test_step_fn = test_step

with open('{}/{}/config.yaml'.format(ROOT, pt_folder)) as fp:
    config = yaml.load(fp, Loader=yaml.FullLoader)

with open('{}/{}/pe_config.yaml'.format(ROOT, pt_folder)) as fp:
    pe_config = yaml.load(fp, Loader=yaml.FullLoader)

for spc in spc_list:
    dict_metrics_datasets = dict()
    for ds_name in ds_names:    

        # To scale de PE freq
        data_to_scale = None
        if scale_pe_freq:
            data_to_scale = ds_name

        astromer = get_ASTROMER(num_layers=config['num_layers'],
                                num_heads=config['num_heads'],
                                head_dim=config['head_dim'],
                                mixer_size=config['mixer'],
                                dropout=config['dropout'],
                                pe_type=config['pe_type'],
                                pe_config=pe_config,
                                pe_func_name=config['pe_func_name'],
                                residual_type=config['residual_type'],
                                window_size=config['window_size'],
                                encoder_mode=config['encoder_mode'],
                                average_layers=config['avg_layers'],
                                data_name=data_to_scale)

        astromer.load_weights(os.path.join(ROOT, pt_folder, 'weights', 'weights'))

        dict_metrics_datasets[ds_name] = dict() 
        for fold in folds:    
            print('Testing {} - fold {}'.format(ds_name.upper(), fold))

            if isinstance(spc, str):
                path_data = './data/records/{}/fold_{}/{}'.format(ds_name, fold, ds_name)
                path_save = os.path.join(ROOT, pt_folder, 'test_metrics_datasets')
            else:
                path_data = './data/records/{}/fold_{}/{}_{}'.format(ds_name, fold, ds_name, spc)
                path_save = os.path.join(ROOT, pt_folder, 'test_metrics_datasets_{}'.format(spc))

            if scale_pe_freq:
                path_save += path_save + '_pe_by_mean'

            test_loader = pretraining_pipeline(os.path.join(path_data, 'test'),
                                                batch_size=5 if config['debug'] else config['bs'],
                                                window_size=config['window_size'],
                                                shuffle=False,
                                                sampling=False,
                                                msk_frac=config['probed'],
                                                rnd_frac=config['rs'],
                                                same_frac=config['rs'],
                                                key_format='1')

            test_logs = []
            for x, y in test_loader:
                logs = test_step_fn(astromer, x, y)
                test_logs.append(logs)

            test_metrics = average_logs(test_logs)
            dict_metrics_datasets[ds_name]['fold_{}'.format(fold)] = test_metrics

    with open('{}.toml'.format(path_save), 'w') as fp:
        toml.dump(dict_metrics_datasets, fp)

