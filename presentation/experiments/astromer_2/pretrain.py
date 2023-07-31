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
from datetime import datetime

ROOT = './presentation/experiments/astromer_2/'

def run(opt):
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

	EXPDIR = os.path.join(ROOT, 'results', opt.exp_name)
	os.makedirs(EXPDIR, exist_ok=True)

	# ========== DATA ========================================
	train_batches = load_data(dataset=os.path.join(opt.data, 'train'), 
							  batch_size=opt.bs, 
							  probed=opt.probed,
							  random_same=opt.rs,  
							  window_size=opt.ws, 
							  nsp_prob=opt.nsp_prob, 
							  repeat=4, 
							  sampling=True,
							  off_nsp=opt.off_nsp)
	valid_batches = load_data(dataset=os.path.join(opt.data, 'val'), 
							  batch_size=opt.bs, 
							  probed=opt.probed,  
							  random_same=opt.rs,
							  window_size=opt.ws, 
							  nsp_prob=opt.nsp_prob, 
							  repeat=1, 
							  sampling=True,
							  off_nsp=opt.off_nsp)

	if opt.nsp_prob == 0. or opt.off_nsp:
		opt.rmse_factor = 1.

	# ========== DEBUG ======================================
	if opt.debug:
		print('[INFO] DEBGUGING MODE')
		train_batches  = train_batches.take(2)
		valid_batches  = valid_batches.take(2)
		opt.epochs = 2

	# ======= MODEL ========================================
	now = datetime.now()
	model_name = now.strftime("%Y-%m-%d_%H-%M-%S")
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
							   pe_dim=opt.pe_dim,
							   pe_c=1,
							   window_size=opt.ws,
							   encoder_mode=opt.encoder_mode,
							   average_layers=opt.avg_layers,
							   off_nsp=opt.off_nsp)
	if opt.optimizer == 'adam':
		optimizer = Adam(opt.lr)
	if opt.optimizer == 'adamw':
		optimizer = AdamW(opt.lr)

	bce_factor    = 1.- opt.rmse_factor
	astromer.compile(rmse_factor=opt.rmse_factor, optimizer=optimizer)

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
	print("[INFO] LOGS: ", os.path.join(EXPDIR, model_name) )
	print(f'[INFO] AVG LAYERS: {opt.avg_layers}')
	print(f'[INFO] NSP OFF: {opt.off_nsp}')
	print(f'[INFO] ENCODER: {opt.encoder_mode}')
	print('[INFO] BCE: {:.2f} RMSE: {:.2f}'.format(bce_factor, opt.rmse_factor))
	print(f'[INFO] No LAYERS: {opt.layers}')
	print(f'[INFO] No HEADS: {opt.nh}')
	print(f'[INFO] HEAD DIM: {opt.hdim}')
	print(f'[INFO] USING {str(opt.optimizer).upper()} OPTIMIZER')
	print('\n')

	start = time.time()
	hist = astromer.fit(train_batches, 
						epochs=opt.epochs, 
						validation_data=valid_batches,
						callbacks=callbacks)      
	training_time = time.time() - start

	min_index = tf.argmin(hist.history['val_loss'])

	# ======== TESTING =========================================
	test_batches = load_data(dataset=os.path.join(opt.data, 'test'), 
							  batch_size=opt.bs, 
							  probed=opt.probed,  
							  random_same=opt.rs,
							  window_size=opt.ws, 
							  nsp_prob=opt.nsp_prob, 
							  repeat=1, 
							  sampling=True,
							  off_nsp=opt.off_nsp)
	
	if opt.debug:
		test_batches = test_batches.take(1)
	
	if opt.off_nsp:
		loss, r2 = astromer.evaluate(test_batches) 
		with open(os.path.join(PTWEIGTHS, 'metrics.toml'), 'w') as fp:
			toml.dump({'data':os.path.join(opt.data, 'test'),
					   'training_time_sec': training_time,
					   'val_rmse': float(hist.history['val_loss'][min_index]),
					   'val_r2': float(hist.history['val_r_square'][min_index]),
					   'train_rmse': float(hist.history['loss'][min_index]),
					   'r2': float(hist.history['r_square'][min_index]),                   
					   'test_rmse':loss, 
					   'test_r2':r2}, fp)
	else:
		acc, bce, loss, r2, rmse = astromer.evaluate(test_batches)   
		with open(os.path.join(PTWEIGTHS, 'metrics.toml'), 'w') as fp:
			toml.dump({'data':os.path.join(opt.data, 'test'),
					   'training_time_sec': training_time,
					   'val_rmse': float(hist.history['val_rmse'][min_index]),
					   'val_r2': float(hist.history['val_r_square'][min_index]),
					   'val_bce': float(hist.history['val_bce'][min_index]),
					   'val_acc': float(hist.history['val_acc'][min_index]),
					   'train_bce': float(hist.history['bce'][min_index]),
					   'train_acc': float(hist.history['acc'][min_index]),
					   'train_rmse': float(hist.history['rmse'][min_index]),
					   'r2': float(hist.history['r_square'][min_index]),                   
					   'test_acc': acc, 
					   'test_r2':r2, 
					   'test_rmse':rmse, 
					   'test_bce':bce}, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default='pretrain', type=str,
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
    parser.add_argument('--pe-dim', default=256, type=int,
                        help='Positional encoder size - i.e., Number of frequencies')
    parser.add_argument('--mixer', default=256, type=int,
                        help='Units to be used on the hidden layer of a feed-forward network that combines head outputs within an attention layer')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Dropout to use on the output of each attention layer (before mixer layer)')
    parser.add_argument('--avg-layers', action='store_true', help='If averaging outputs of the attention layers to form the final embedding. There is no avg if layers=1 ')

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
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='adam, adamw')

    parser.add_argument('--off-nsp',  action='store_true', help='Turn off the NSP input format (use to load Astromer I)')
    parser.add_argument('--probed', default=0.5, type=float,
                        help='Probed percentage')
    parser.add_argument('--rs', default=0.2, type=float,
                        help='Probed fraction to be randomized or unmasked')
    parser.add_argument('--nsp-prob', default=0.5, type=float,
                        help='Next segment prediction probability. The probability of randomize half of the light curve')
    parser.add_argument('--rmse-factor', default=0.5, type=float,
                        help='RMSE weight factor. The loss function will be loss = rmse_factor*rmse + (1 - rmse_factor)*bce')


    opt = parser.parse_args()        
    run(opt)
