import tensorflow as tf 
import pandas as pd
import argparse
import toml
import time
import sys
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.callbacks  import (ModelCheckpoint,
										 EarlyStopping,
										 TensorBoard)
from src.models import get_ASTROMER_II
from src.data import load_data


ROOT = './presentation/experiments/astromer_2/'

def run(opt):
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

	EXPDIR = os.path.join(ROOT, 'results', opt.exp_name)
	os.makedirs(EXPDIR, exist_ok=True)

	# ========== DATA ========================================
	train_batches = load_data(dataset=os.path.join(opt.data, 'train'), 
							  batch_size=opt.bs, 
							  probed=opt.probed,  
							  window_size=opt.ws, 
							  nsp_prob=opt.nsp_prob, 
							  repeat=4, 
							  sampling=True)
	valid_batches = load_data(dataset=os.path.join(opt.data, 'val'), 
							  batch_size=opt.bs, 
							  probed=opt.probed,  
							  window_size=opt.ws, 
							  nsp_prob=opt.nsp_prob, 
							  repeat=1, 
							  sampling=True)
	# ========== DEBUG ======================================
	if opt.debug:
		print('[INFO] DEBGUGING MODE')
		train_batches  = train_batches.take(2)
		valid_batches  = valid_batches.take(2)
		opt.epochs = 5

	# ======= MODEL ========================================
	model_name = '{}_{}_{}_rmse_{}'.format(opt.layers, opt.nh, opt.hdim, opt.rmse_factor)
	PTWEIGTHS = os.path.join(EXPDIR, model_name, 'pretraining')
	os.makedirs(PTWEIGTHS, exist_ok=True)

	with open(os.path.join(PTWEIGTHS, 'config.toml'), 'w') as f:
		toml.dump(opt.__dict__, f)

	astromer = get_ASTROMER_II(num_layers=opt.layers,
							   num_heads=opt.nh,
							   head_dim=opt.hdim,
							   mixer_size=opt.mixer,
							   dropout=opt.dropout,
							   pe_base=1000,
							   pe_dim=128,
							   pe_c=1,
							   window_size=opt.ws,
							   encoder_mode=opt.encoder_mode)

	optimizer = AdamW(opt.lr)
	bce_factor    = 1.- opt.rmse_factor
	astromer.compile(rmse_factor=opt.rmse_factor, bce_factor=bce_factor, optimizer=optimizer)

	callbacks = [
			ModelCheckpoint(
				filepath=os.path.join(PTWEIGTHS, 'weights'),
				save_weights_only=True,
				monitor='val_loss',
				save_best_only=True),
			EarlyStopping(monitor='val_loss',
				patience = opt.patience,
				restore_best_weights=True),
			TensorBoard(
				log_dir = os.path.join(PTWEIGTHS, 'logs'),
				histogram_freq=1,
				write_graph=True)]

	print('\n')
	print(f'[INFO] ENCODER: {opt.encoder_mode}')
	print('[INFO] BCE: {:.2f} RMSE: {:.2f}'.format(bce_factor, opt.rmse_factor))
	print(f'[INFO] No LAYERS: {opt.layers}')
	print(f'[INFO] No HEADS: {opt.nh}')
	print(f'[INFO] HEAD DIM: {opt.hdim}')
	print('\n')

	start = time.time()
	hist = astromer.fit(train_batches, 
						epochs=opt.epochs, 
						validation_data=valid_batches,
						callbacks=callbacks)      
	training_time = time.time() - start

	# ======== TESTING =========================================
	test_batches = load_data(dataset=os.path.join(opt.data, 'test'), 
							  batch_size=opt.bs, 
							  probed=opt.probed,  
							  window_size=opt.ws, 
							  nsp_prob=opt.nsp_prob, 
							  repeat=1, 
							  sampling=True)
	if opt.debug:
		test_batches = test_batches.take(1)
	
	acc, bce, loss, r2, rmse = astromer.evaluate(test_batches)   
	with open(os.path.join(PTWEIGTHS, 'metrics.toml'), 'w') as fp:
		toml.dump({'data':os.path.join(opt.data, 'test'),
				   'training_time_sec': training_time,
				   'val_rmse': float(tf.reduce_min(hist.history['val_rmse']).numpy()),
				   'val_r2': float(tf.reduce_min(hist.history['val_r_square']).numpy()),
				   'val_bce': float(tf.reduce_min(hist.history['val_bce']).numpy()),
				   'val_acc': float(tf.reduce_min(hist.history['val_acc']).numpy()),
				   'train_bce': float(tf.reduce_min(hist.history['bce']).numpy()),
				   'train_acc': float(tf.reduce_min(hist.history['acc']).numpy()),
				   'train_rmse': float(tf.reduce_min(hist.history['rmse']).numpy()),
				   'r2': float(tf.reduce_min(hist.history['r_square']).numpy()),                   
				   'test_acc': acc, 
				   'test_r2':r2, 
				   'test_rmse':rmse, 
				   'test_bce':bce}, fp)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp-name', default='nsp_adamw_factor', type=str,
					help='Project name')
	parser.add_argument('--data', default='./data/records/macho_clean', type=str,
					help='Data folder where tf.record files are located')
	parser.add_argument('--gpu', default='-1', type=str,
						help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


	parser.add_argument('--encoder-mode', default='normal', type=str,
						help='normal - conditioned')
	parser.add_argument('--layers', default=1, type=int,
						help='Number of Attention Layers')
	parser.add_argument('--nh', default=4, type=int,
						help='Number of heads within the attention layer')
	parser.add_argument('--hdim', default=64, type=int,
						help='Head dimension')
	parser.add_argument('--mixer', default=256, type=int,
						help='Units to be used on the hidden layer of a feed-forward network that combines head outputs within an attention layer')
	parser.add_argument('--dropout', default=0.1, type=float,
						help='Dropout to use on the output of each attention layer (before mixer layer)')

	parser.add_argument('--lr', default=1e-5, type=float,
						help='learning rate')
	parser.add_argument('--bs', default=16, type=int,
						help='Batch size')
	parser.add_argument('--patience', default=20, type=int,
						help='Earlystopping threshold in number of epochs')
	parser.add_argument('--epochs', default=100000, type=int,
						help='Number of epochs')
	parser.add_argument('--ws', default=200, type=int,
						help='windows size of the PSFs')

	parser.add_argument('--probed', default=0.5, type=float,
						help='Probed percentage')
	parser.add_argument('--nsp-prob', default=0.5, type=float,
						help='Next segment prediction probability. The probability of randomize half of the light curve')
	parser.add_argument('--rmse-factor', default=0.5, type=float,
						help='RMSE weight factor. The loss function will be loss = rmse_factor*rmse + (1 - rmse_factor)*bce')


	opt = parser.parse_args()        
	run(opt)
