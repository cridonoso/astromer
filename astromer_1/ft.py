import tensorflow as tf
import pandas as pd
import argparse
import toml
import sys
import os
import pickle


from presentation.experiments.utils import train_classifier
from src.models.astromer_1 import get_ASTROMER, build_input, train_step, test_step
from src.training.utils import train
from src.data import load_data
from datetime import datetime

from tensorflow.keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import LayerNormalization, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def merge_metrics(**kwargs):
	merged = {}
	for key, value in kwargs.items():
		for subkey, subvalue in value.items():
			merged['{}_{}'.format(key, subkey)] = subvalue
	return merged


def run(opt):
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
	# ====================================================================================
	# =============== LOADING PRETRAINED MODEL ===========================================
	# ====================================================================================
	with open(os.path.join(opt.pt_folder, 'model_config.toml'), 'r') as f:
		model_config = toml.load(f)

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
							average_layers=model_config['avg_layers'],
							batch_size=2500)

	astromer.load_weights(os.path.join(opt.pt_folder, 'fold_0/weights', 'weights'))
	print('[INFO] Weights loaded')
	# ====================================================================================
	# =============== FINETUNING MODEL  ==================================================
	# ====================================================================================
	DOWNSTREAM_DATA = os.path.join('./records', 
							   opt.subdataset,
							   'fold_'+str(opt.fold), 
							   '{}_{}'.format(opt.subdataset, opt.spc))
	FTWEIGTHS = os.path.join(opt.pt_folder,
							 'finetuning',                                     
							 opt.subdataset,
							 'fold_'+str(opt.fold), 
							 '{}_{}'.format(opt.subdataset, opt.spc)) 
							 
	train_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'train'), 
							 random_same=model_config['rs'],  
							 probed=model_config['probed'],
							 off_nsp=True, 
							 window_size=model_config['window_size'], 
							 sampling=False,
							 repeat=1, 
							 batch_size= opt.bs) 
	valid_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'val'), 
							 random_same=model_config['rs'],
							 probed=model_config['probed'],  
							 off_nsp=True,
							 window_size=model_config['window_size'], 
							 sampling=False,
							 repeat=1, 
							 batch_size= opt.bs)
	test_loader = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'test'), 
							 batch_size=  opt.bs, 
							 random_same=model_config['rs'],
							 probed=model_config['probed'],  
							 off_nsp=True,
							 window_size=model_config['window_size'], 
							 repeat=1, 
							 sampling=False)
	astromer, \
	(best_train_metrics,
	best_val_metrics,
	test_metrics), callbacks  = train(astromer,
						   train_loader, 
						   valid_loader, 
						   num_epochs=opt.num_epochs, 
						   lr=model_config['lr'], 
						   test_loader=test_loader,
						   project_path=FTWEIGTHS,
						   debug=False,
						   patience=opt.patience,
						   train_step_fn=train_step,
						   test_step_fn=test_step,
						   argparse_dict=opt.__dict__)

	metrics = merge_metrics(train=best_train_metrics, 
							val=best_val_metrics, 
							test=test_metrics)

	with open(os.path.join(FTWEIGTHS, 'metrics.toml'), 'w') as fp:
		toml.dump(metrics, fp)

	with open(os.path.join(FTWEIGTHS, 'config.toml'), 'w') as f:
		toml.dump(model_config, f)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='-1', type=str, help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--subdataset', default='alcock', type=str, help='Data folder where tf.record files are located')
	parser.add_argument('--pt-folder', default='./results/pretraining*', type=str, help='pretrained model folder')
	parser.add_argument('--fold', default=0, type=int, help='Fold to use')
	parser.add_argument('--spc', default=20, type=int, help='Samples per class')


	parser.add_argument('--bs', default=2000, type=int,	help='Batch size')
	parser.add_argument('--patience', default=20, type=int,	help='Earlystopping threshold in number of epochs')
	parser.add_argument('--num_epochs', default=10000, type=int, help='Number of epochs')

	opt = parser.parse_args()        
	run(opt)
