''' 

Testing the pretrained model in other datasets 

'''

from src.models.astromer_1 import get_ASTROMER, test_step
from src.data import load_data

import toml
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

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

ds_names = ['alcock', 'ogle', 'atlas']
#ds_names = ['alcock', 'ogle', 'atlas', 'kepler', 'kepler_alcock_linear', 'kepler_atlas_linear', 'kepler_ogle_linear']

folds = [0, 1, 2]
spc_list = [500]
#spc_list = ['all']

ROOT = './presentation/experiments/astromer_1_pe'
pt_folder = 'results/pretraining/P05R02/exp_002'

test_step_fn = test_step

with open('{}/{}/config.toml'.format(ROOT, pt_folder), mode="r") as fp:
    config = toml.load(fp)

with open('{}/{}/pe_config.toml'.format(ROOT, pt_folder), mode="r") as fp:
    pe_config = toml.load(fp)

astromer = get_ASTROMER(num_layers=config['num_layers'],
                        num_heads=config['num_heads'],
                        head_dim=config['head_dim'],
                        mixer_size=config['mixer'],
                        dropout=config['dropout'],
                        pe_type=config['pe_type'],
                        pe_config=pe_config,
                        window_size=config['window_size'],
                        encoder_mode=config['encoder_mode'],
                        average_layers=config['avg_layers'])

astromer.load_weights(os.path.join(ROOT, pt_folder, 'weights', 'weights'))

for spc in spc_list:
    dict_metrics_datasets = dict()
    for ds_name in ds_names:     
        dict_metrics_datasets[ds_name] = dict() 
        for fold in folds:    
            print('Testing {} - fold {}'.format(ds_name.upper(), fold))

            if isinstance(spc, str):
                path_data = './data/records/{}/fold_{}/{}'.format(ds_name, fold, ds_name)
                path_save = os.path.join(ROOT, pt_folder, 'test_metrics_datasets.toml')
            else:
                path_data = './data/records/{}/fold_{}/{}_{}'.format(ds_name, fold, ds_name, spc)
                path_save = os.path.join(ROOT, pt_folder, 'test_metrics_datasets_{}.toml'.format(spc))

            test_loader = load_data(dataset=os.path.join(path_data, 'test'),
                                    batch_size=config['bs'], 
                                    probed=config['probed'], # 1.,  
                                    random_same=config['rs'], #0.,
                                    window_size=config['window_size'], 
                                    off_nsp=True, 
                                    repeat=1, 
                                    sampling=False)

            test_logs = []
            for x, y in test_loader:
                logs = test_step_fn(astromer, x, y)
                test_logs.append(logs)

            test_metrics = average_logs(test_logs)
            dict_metrics_datasets[ds_name]['fold_{}'.format(fold)] = test_metrics

    with open(path_save, 'w') as fp:
        toml.dump(dict_metrics_datasets, fp)

