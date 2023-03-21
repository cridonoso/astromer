'''
HYPERPARAMETER TUNNING USING WANDB 
BY Cristobal 2023
'''
import tensorflow as tf
import pandas as pd
import wandb
import sys
import os

from wandb.keras import WandbMetricsLogger

from src.training import CustomSchedule
from src.models import get_ASTROMER, build_input
from src.data import pretraining_pipeline

from sklearn.metrics import precision_recall_fscore_support

from tensorflow.keras.losses     import CategoricalCrossentropy
from tensorflow.keras.layers 	 import Dense, Conv1D, Flatten
from tensorflow.keras 			 import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] # which gpu to use

sweep_conf = {
	'name': 'my_sweep',
	'method': 'bayes',
	'metric': {'goal': 'maximize', 'name': 'f1'},
	'parameters': {
		'n_layers': {'values':[1, 2, 4, 6, 8]},
		'n_heads': {'values':[1, 2, 4, 8, 16]},
		'head_dim': {'values':[16, 32, 64, 128]},
		'dff': {'values':[16, 32, 64, 128, 256]},
		'dropout_rate': {'distribution':'uniform', 'min':.0, 'max':.5},
	},
	'total_runs': 100
}

def create_classifier(astromer, z_dim, num_cls, n_steps=200, name='mlp_att'):
	placeholder = build_input(n_steps)
	encoder = astromer.get_layer('encoder')
	encoder.trainable = False
	conv_shift = 4
	x = encoder(placeholder, training=False)
	x = Conv1D(32, 5, activation='relu', input_shape=[n_steps, z_dim])(x)
	x = Flatten()(tf.expand_dims(x, 1))
	x = tf.reshape(x, [-1, (n_steps-conv_shift)*32])
	x = Dense(num_cls)(x)
	return Model(inputs=placeholder, outputs=x, name=name)

def get_batchsize(model):
	known_params = 662019
	known_bs  	 = 2000
	return int((known_params*known_bs)/model.count_params())

def main():

	run = wandb.init(project='astromer_0')

	n_layers 	 = wandb.config.n_layers
	n_heads  	 = wandb.config.n_heads
	head_dim 	 = wandb.config.head_dim
	dff  		 = wandb.config.dff
	dropout_rate = wandb.config.dropout_rate

	window_size = 200
	data_path = './data/records/macho/{}'
	d_model = head_dim*n_heads

	astromer =  get_ASTROMER(num_layers=n_layers,
							 d_model=d_model,
							 num_heads=n_heads,
							 dff=dff,
							 base=1000,
							 dropout=dropout_rate,
							 maxlen=window_size,
							 pe_c=1.)
	batch_size = get_batchsize(astromer)
	wandb.log({'batch_size':batch_size})

	lr = CustomSchedule(d_model)
	optimizer = Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
	astromer.compile(optimizer=optimizer)

	data = dict()
	for subset in ['train', 'val']:
		data[subset] = pretraining_pipeline(data_path.format(subset), batch_size, window_size, .5, .2,	.2,
											sampling=True, shuffle=True, repeat=1, num_cls=None,
											normalize=True, cache=True)

	earlystop_callback   = EarlyStopping(monitor='val_loss', patience = 20, restore_best_weights=True),
	# checkpoint_callback  = ModelCheckpoint(filepath=f'./results/hp/trial_{wandb.run.id}/pretraining/model_weights.h5', 
	# 									   save_best_only=True)
	tensorboard_callback = TensorBoard(log_dir=f'./results/hp/trial_{wandb.run.id}/pretraining/logs')
	wandb_callback 		 = WandbMetricsLogger()

	_ = astromer.fit(data['train'],
	 			 		epochs=10000,
				 		validation_data=data['val'], 
				 		callbacks=[earlystop_callback, 
				 			# checkpoint_callback, 
				 			tensorboard_callback, 
				 			wandb_callback])
       
	# ==========================================================================================
	# ======= CLASSIFICATION ===================================================================
	# ==========================================================================================
	data = dict()
	fold_n = 1
	for subset in ['train', 'val']:
		num_cls = pd.read_csv(
			os.path.join(f'./data/records/alcock/fold_{fold_n}/alcock_50/', 'objects.csv')).shape[0]

		data[subset] = pretraining_pipeline(
			f'./data/records/alcock/fold_{fold_n}/alcock_50/{subset}',
			512, 200, 0., 0., 0., False,	True, repeat=1,	num_cls=num_cls,
			normalize=True, cache=True)

	clf_model = create_classifier(astromer, d_model, num_cls, n_steps=window_size, name='mlp_att')
	optimizer = Adam(learning_rate=1e-3)
	clf_model.compile(optimizer=optimizer,
					  loss=CategoricalCrossentropy(from_logits=True),
					  metrics='accuracy')

	earlystop_callback   = EarlyStopping(monitor='val_loss', patience = 20, restore_best_weights=True),
	tensorboard_callback = TensorBoard(log_dir=f'./results/hp/trial_{wandb.run.id}/classification')

	_ = clf_model.fit(data['train'],
					  epochs=1000,
					  callbacks=[earlystop_callback,
								 tensorboard_callback],
					  validation_data=data['val'])

	y_pred = clf_model.predict(data['val'])
	y_true = tf.concat([y for _, y in data['val']], 0)

	pred_labels = tf.argmax(y_pred, 1)
	true_labels = tf.argmax(y_true, 1)

	p, r, f, _ = precision_recall_fscore_support(true_labels,
	                                             pred_labels,
	                                             average='macro',
	                                             zero_division=0.)

	wandb.log({'f1':f})

sweep_id = wandb.sweep(sweep=sweep_conf, project='astromer_0')
wandb.agent(sweep_id,
			function=main, 
			count=1)