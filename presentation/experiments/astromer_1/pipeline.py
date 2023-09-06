import tensorflow as tf 
import pandas as pd
import argparse
import wandb
import glob
import toml
import os

from presentation.experiments.utils import train_classifier
from wandb.keras import WandbMetricsLogger

from src.models.astromer_1 import get_ASTROMER, build_input, train_step, test_step
from src.training.utils import train
from src.data import load_data

tf.config.run_functions_eagerly(True)

def adjust_fn(func, **karguments):
	def wrap(*args, **kwargs):
		result = func(*args, **karguments)
		return result
	return wrap

def check_if_exist_finetuned_weights(config, project_name):
	api = wandb.Api()
	runs = api.runs(project_name)
	for run in runs:
		if run.config['subdataset'] == config.subdataset and \
		   run.config['fold']== config.fold and \
		   run.state == 'finished':
			return True
	return False
	
def merge_metrics(**kwargs):
	merged = {}
	for key, value in kwargs.items():
		for subkey, subvalue in value.items():
			merged['{}_{}'.format(key, subkey)] = subvalue
	return merged

def sweep_train(config=None, opt=None):
	with wandb.init(config=config):
		config = wandb.config                   

		# ====================================================================================
		# PRE-TRAINING =======================================================================
		# ====================================================================================

		with open(os.path.join(config.pt_model, 'config.toml'), 'r') as f:
			model_config = toml.load(f)
			wandb.log(model_config)


		astromer = get_ASTROMER(num_layers=model_config['num_layers'],
								num_heads=model_config['num_heads'],
								head_dim=model_config['head_dim'],
								mixer_size=model_config['mixer'],
								dropout=model_config['dropout'],
								pe_base=model_config['pe_base'],
								pe_dim=model_config['pe_dim'],
								pe_c=model_config['pe_exp'],
								window_size=model_config['window_size'],
								encoder_mode=model_config['encoder_mode'],
								average_layers=model_config['avg_layers'])

		print('[INFO] LOADING PRETRAINED WEIGHTS')
		astromer.load_weights(os.path.join(config.pt_model, 'weights', 'weights'))

		# =====================================================================================
		# === FINETUNING STEP =================================================================
		# =====================================================================================  
		DOWNSTREAM_DATA = os.path.join('./data/records', 
									   config.subdataset,
									   'fold_'+str(config.fold), 
									   '{}_{}'.format(config.subdataset, config.spc))
		FTWEIGTHS = os.path.join(config.pt_model,
								 '..',
								 'finetuning',                                     
								 config.subdataset,
								 'fold_'+str(config.fold), 
								 '{}_{}'.format(config.subdataset, config.spc))     

		does_it_exists = check_if_exist_finetuned_weights(config, opt.wandb_name)

		if os.path.isfile(os.path.join(FTWEIGTHS, 'weights', 'checkpoint')) and does_it_exists:
			print('[INFO] FINETUNED MODEL FOUND. LOADING WEIGHTS')
			astromer.load_weights(os.path.join(FTWEIGTHS, 'weights', 'weights')).expect_partial()
			with open(os.path.join(FTWEIGTHS, 'metrics.toml'), 'r') as fp:
				metrics = toml.load(fp)
		else:
			print('[INFO] FINETUNING FROM SCRATCH')
			train_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'train'), 
									 batch_size= 5 if opt.debug else model_config['bs'], 
									 probed=model_config['probed'],
									 random_same=model_config['rs'],  
									 window_size=model_config['window_size'], 
									 off_nsp=True, 
									 repeat=1, 
									 sampling=False)
			valid_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'val'), 
									 batch_size= 5 if opt.debug else model_config['bs'], 
									 probed=model_config['probed'],  
									 random_same=model_config['rs'],
									 window_size=model_config['window_size'], 
									 off_nsp=True,
									 repeat=1, 
									 sampling=False)
			test_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'test'), 
									 batch_size= 5 if opt.debug else model_config['bs'], 
									 probed=model_config['probed'],  
									 random_same=model_config['rs'],
									 window_size=model_config['window_size'], 
									 off_nsp=True,
									 repeat=1, 
									 sampling=False)

			astromer, \
			(best_train_metrics,
			best_val_metrics,
			test_metrics) = train(astromer,
								 train_loader, 
								 valid_loader, 
								 num_epochs=model_config['num_epochs'], 
								 lr=model_config['lr'], 
								 test_loader=test_loader,
								 project_path=FTWEIGTHS,
								 debug=opt.debug,
								 patience=model_config['patience'],
								 train_step_fn=train_step,
								 test_step_fn=test_step,
								 argparse_dict=model_config)
			metrics = merge_metrics(train=best_train_metrics, 
									val=best_val_metrics, 
									test=test_metrics)

			with open(os.path.join(FTWEIGTHS, 'metrics.toml'), 'w') as fp:
				toml.dump(metrics, fp)

			with open(os.path.join(FTWEIGTHS, 'config.toml'), 'w') as f:
				toml.dump(model_config, f)

		wandb.log(metrics)
		# ============================================================================
		# =========== CLASSIFICATION==================================================
		# ============================================================================
		CLFWEIGHTS = os.path.join( config.pt_model,
								  '..',
								  'classification', 
								  config.subdataset, 
								  'fold_'+str(config.fold), 
								  config.subdataset+'_20')
		

		num_cls = pd.read_csv(os.path.join(DOWNSTREAM_DATA, 'objects.csv')).shape[0]

		train_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'train'), 
								 batch_size= 5 if opt.debug else model_config['bs'], 
								 probed=1.,
								 random_same=0.,  
								 window_size=model_config['window_size'], 
								 off_nsp=True,
								 nsp_prob=0., 
								 repeat=1, 
								 sampling=False,
								 shuffle=True,
								 num_cls=num_cls)
		valid_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'val'), 
								 batch_size= 5 if opt.debug else model_config['bs'], 
								 probed=1.,
								 random_same=0.,  
								 window_size=model_config['window_size'], 
								 off_nsp=True,
								 nsp_prob=0., 
								 repeat=1, 
								 sampling=False,
								 num_cls=num_cls)
		test_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'test'), 
								 batch_size= 5 if opt.debug else model_config['bs'], 
								 probed=1.,
								 random_same=0.,  
								 window_size=model_config['window_size'], 
								 off_nsp=True, 
								 nsp_prob=0., 
								 repeat=1, 
								 sampling=False,
								 num_cls=num_cls)

		if opt.debug:
			train_loader = train_loader.take(1)
			valid_loader = valid_loader.take(1)
			test_loader  = test_loader.take(1)

		# First Layer
		inp_placeholder = build_input(model_config['window_size'])
		encoder = astromer.get_layer('encoder')
		encoder.trainable = opt.train_astromer
		embedding = encoder(inp_placeholder)
		embedding = embedding*(1.-inp_placeholder['att_mask'])
		embedding = tf.math.divide_no_nan(tf.reduce_sum(embedding, axis=1), 
										  tf.reduce_sum(1.-inp_placeholder['att_mask']))
		
		summary_clf = train_classifier(embedding,
									   inp_placeholder=inp_placeholder,
									   train_loader=train_loader,
									   valid_loader=valid_loader, 
									   test_loader=test_loader,
									   num_cls=num_cls, 
									   project_path=CLFWEIGHTS,
									   clf_name=config.clf_name,
									   debug=opt.debug)

		wandb.log(summary_clf)


if __name__ == '__main__':   
	parser = argparse.ArgumentParser()
	parser.add_argument('--pt-folder', default='./presentation/experiments/astromer_2/results/*', type=str,
					help='Pretraining folder(s) - use * to jump folders up to "pretraining" folder is found')
	parser.add_argument('--wandb_name', default='astromer_1_exp', type=str,
					help='Table name used in wandb')
	parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')
	parser.add_argument('--train-astromer', action='store_true', help='train encoder when classifying')
	parser.add_argument('--sweep-id', default='', type=str,
						help='if 0 then a new sweep is created on wandb')
	parser.add_argument('--gpu', default='-1', type=str,
						help='GPU to be used. -1 means no GPU will be used')
	opt = parser.parse_args()   
	
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

	folder_names = glob.glob(os.path.join(opt.pt_folder, 'pretraining'))


	sweep_conf = {
		'name': 'ASTROMER_II',
		'method': 'grid',
		'metric': {'goal': 'maximize', 'name': 'val_acc'},
		'parameters': {
			'pt_model': {'values':folder_names},
			'fold':{'values':[0, 1, 2]},
			'spc': {'values': [20, 100]},
			'subdataset':{'values':['atlas', 'alcock']},
			'clf_name':{'values':['att_mlp']}, 
		}
	}

	sweep_train_ = adjust_fn(sweep_train, opt=opt)

	sweep_id = wandb.sweep(sweep_conf, project=opt.wandb_name)
	if opt.sweep_id != '':
		sweep_id = opt.sweep_id

	wandb.agent(sweep_id, function=sweep_train_, count=100)