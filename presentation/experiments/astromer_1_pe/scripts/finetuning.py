import pandas as pd
import argparse
import toml
import yaml
import os

from src.models.astromer_1 import get_ASTROMER, train_step, test_step
from src.training.utils import train
from src.data.zero import pretraining_pipeline


def maybe_str_or_int(arg):
    try:
        return int(arg)  # try convert to int
    except ValueError:
        pass
    if arg == "all":
        return arg
    raise argparse.ArgumentTypeError("x must be an int or 'all'")


def get_none_values(model_config):
	if 'residual_type' not in model_config:
		model_config['residual_type'] = None
	return model_config


def set_trainable_layers(model, ft_science_case, layers_config):
	print(f'Processing science_case: {ft_science_case}...')

	for layer in model.get_layer('encoder').layers:
		print(f'Name layers: {layer.name}')
		print('ft_science_case: {}'.format(ft_science_case))
		if ft_science_case.find('ATT1') != -1 and ft_science_case.find('ATT2') != -1:
			if layer.name == 'inp_transform':  
				layer.trainable = layers_config[ft_science_case]['ff1_layer']
			elif layer.name.find('att_') != -1:
				layer.trainable = layers_config[ft_science_case]['att_layer']

		else:
			if layer.name == 'inp_transform':  
				layer.trainable = layers_config[ft_science_case]['ff1_layer']
			elif layer.name == 'att_layer_0':
				layer.trainable = layers_config[ft_science_case]['att_layer_1']
			elif layer.name == 'att_layer_1':
				layer.trainable = layers_config[ft_science_case]['att_layer_2']

	for layer in model.layers:
		if layer.name != 'encoder':
			layer.trainable = layers_config[ft_science_case]['ff2_layer']

	return model


def merge_metrics(**kwargs):
	merged = {}
	for key, value in kwargs.items():
		for subkey, subvalue in value.items():
			merged['{}_{}'.format(key, subkey)] = subvalue
	return merged


def run(opt):
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

	ROOT = './presentation/experiments/astromer_1_pe/'

	if isinstance(opt.spc, str):
		data_name = opt.dataset
	else:
		data_name = '{}_{}'.format(opt.dataset, opt.spc)

	# To scale de PE freq
	data_to_scale = None
	if opt.scale_pe_freq:
		data_to_scale = opt.dataset

	# =========================================================================
	# =============== LOADING SETTING =========================================
	# ========================================================================+

	# General config for frezee and trainable layers
	with open('./{}/layers_config.toml'.format(ROOT), mode="r") as f:
		layers_config = toml.load(f)

	# Pretraining configs
	file = open(os.path.join(ROOT, opt.pt_folder, 'config.yaml'), "r")
	model_config = yaml.load(file, Loader=yaml.FullLoader) 

	# Update model config
	dict_model_config = dict({
		'Pretraining': model_config.copy(),
		'Finetuning': opt.__dict__
	})

	#if opt.pe_func_name == 'same':
	file = open(os.path.join(ROOT, opt.pt_folder, 'pe_config.yaml'), "r")
	pe_config = yaml.load(file, Loader=yaml.FullLoader) 
	#else:
	#	file = open('{}/pe_config.yaml'.format(ROOT), "r")
	#	pe_config = yaml.load(file, Loader=yaml.FullLoader) 
	#	model_config['pe_func_name'] = opt.pe_func_name


	#with open(os.path.join(ROOT, opt.pt_folder, 'config.toml'), 'r') as f:
	#	model_config = toml.load(f)
	#
	#model_config = get_none_values(model_config)

	#with open(os.path.join(ROOT, opt.pt_folder, 'pe_config.toml'), 'r') as f:
	#	pe_config = toml.load(f)

	# Update PE config
	dict_pe = dict({model_config['pe_func_name']: pe_config[model_config['pe_func_name']]})
	dict_pe[model_config['pe_func_name']]['pe_trainable'] = layers_config[opt.ft_science_case]['pe_layer']

	# ====================================================================================
	# =============== DATA LOAD  =========================================================
	# ====================================================================================
	FINETUNING_DATA = os.path.join('./data/records', 
							   	   opt.dataset,
							   	   'fold_'+str(opt.fold), 
							   	   '{}'.format(data_name)) 

	train_loader = pretraining_pipeline(os.path.join(FINETUNING_DATA, 'train'),
										batch_size=5 if opt.debug else opt.bs, 
										window_size=model_config['window_size'],
										shuffle=True,
										sampling=False,
										repeat=1,
										msk_frac=model_config['probed'],
										rnd_frac=model_config['rs'],
										same_frac=model_config['rs'],
										key_format='1')
	valid_loader = pretraining_pipeline(os.path.join(FINETUNING_DATA, 'val'),
										batch_size=5 if opt.debug else opt.bs,
										window_size=model_config['window_size'],
										shuffle=False,
										sampling=False,
										msk_frac=model_config['probed'],
										rnd_frac=model_config['rs'],
										same_frac=model_config['rs'],
										key_format='1')
	test_loader = pretraining_pipeline(os.path.join(FINETUNING_DATA, 'test'),
										batch_size=5 if opt.debug else opt.bs,
										window_size=model_config['window_size'],
										shuffle=False,
										sampling=False,
										msk_frac=model_config['probed'],
										rnd_frac=model_config['rs'],
										same_frac=model_config['rs'],
										key_format='1')

	# ====================================================================================
	# =============== LOADING PRETRAINED MODEL ===========================================
	# ====================================================================================

	astromer = get_ASTROMER(num_layers=model_config['num_layers'],
							num_heads=model_config['num_heads'],
							head_dim=model_config['head_dim'],
							mixer_size=model_config['mixer'],
							dropout=model_config['dropout'],
							pe_type=model_config['pe_type'],
							pe_config=dict_pe,
                            pe_func_name=model_config['pe_func_name'],
                            residual_type=model_config['residual_type'],
							window_size=model_config['window_size'],
							encoder_mode=model_config['encoder_mode'],
							average_layers=model_config['avg_layers'],
							data_name=data_to_scale)

	astromer.load_weights(os.path.join(ROOT, opt.pt_folder, 'weights', 'weights'))
	print('[INFO] Weights loaded')

	# ====================================================================================
	# =============== FINETUNING MODEL  ==================================================
	# ====================================================================================
	FTWEIGTHS = os.path.join(ROOT,
						  	 opt.ft_folder, 
							 opt.ft_science_case, 
							 opt.dataset,
							 'fold_'+str(opt.fold), 
							 '{}'.format(data_name)) 

	os.makedirs(FTWEIGTHS, exist_ok=True)

	# Trainable and frezee layers
	astromer = set_trainable_layers(astromer, opt.ft_science_case, layers_config)

	print(astromer.summary())
	with open('{}/model_summary.txt'.format(FTWEIGTHS), 'w') as f:
		astromer.summary(print_fn=lambda x: f.write(x + '\n'))
			
	# Save PE configuration
	file = open(os.path.join(FTWEIGTHS, 'pe_config.yaml'), "w")
	yaml.dump(dict_pe, file)
	file.close()

	#with open(os.path.join(FTWEIGTHS, 'pe_config.toml'), 'w') as f:
	#	toml.dump(dict_pe, f)

	# Training
	astromer, \
	(best_train_metrics,
	best_val_metrics)  = train(astromer,
							train_loader, 
							valid_loader, 
							num_epochs=opt.num_epochs, 
							lr=opt.lr, 
							test_loader=test_loader,
							project_path=FTWEIGTHS,
							debug=opt.debug,
							patience=opt.patience,
							train_step_fn=train_step,
							test_step_fn=test_step,
							argparse_dict=dict_model_config)

	metrics = merge_metrics(train=best_train_metrics, 
							val=best_val_metrics)

	with open(os.path.join(FTWEIGTHS, 'train_val_metrics.toml'), 'w') as fp:
		toml.dump(metrics, fp)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='0', type=str, help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--dataset', default='alcock', type=str, help='Data folder where tf.record files are located')
	parser.add_argument('--fold', default=0, type=int, help='Fold to use')
	parser.add_argument('--spc', default=500, type=maybe_str_or_int, help='Samples per class')

	parser.add_argument('--pt-folder', default='results/pretraining/P02R01/macho_pe_nontrainable-2023-09-18_01-12-37', 
						type=str, help='pretrained model folder')

	parser.add_argument('--ft-folder', default='results/finetuning/P02R01/macho_pe_nontrainable_prueba', 
						type=str, help='pretrained model folder')
	parser.add_argument('--ft-science-case', default='FF1_ATT_FF2', type=str, help='Layers trainables')
	parser.add_argument('--scale-pe-freq', action='store_true', help='a debugging flag to be used when testing.')
	parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')

	parser.add_argument('--lr', default='scheduler', type=str, help='learning rate')
	parser.add_argument('--bs', default=2000, type=int,	help='Batch size')
	parser.add_argument('--patience', default=20, type=int,	help='Earlystopping threshold in number of epochs')
	parser.add_argument('--num-epochs', default=10000, type=int, help='Number of epochs')

	parser.add_argument('--pe-func-name', default='same', type=str,
						help='You can select: ["not_pe_module", "use_t", "pe", "pe_mlp", "pe_rnn", "pe_tm", "pe_att"]')

	opt = parser.parse_args()        
	run(opt)
