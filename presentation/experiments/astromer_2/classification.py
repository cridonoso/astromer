import tensorflow as tf
import pandas as pd
import argparse
import toml
import os


from src.layers.downstream import get_astromer_encoder, load_classification_data
from src.models.second import TransformLayer

from tensorflow.keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, LayerNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, Model

from sklearn.metrics import precision_recall_fscore_support


def get_callbacks(path, patience=20, monitor='val_loss'):
	callbacks = [
		ModelCheckpoint(
			filepath=os.path.join(path, 'weights'),
			save_weights_only=True,
			 mode='min',
			monitor=monitor,
			save_best_only=True),
		EarlyStopping(monitor=monitor,
			patience = patience,
			 mode='min',
			restore_best_weights=True),
		TensorBoard(
			log_dir = os.path.join(path, 'logs'),
			histogram_freq=1,
			write_graph=True)]
	return callbacks


def create_classifier(pretrain_weights, window_size, n_classes, clf_name='mlp_att', trainable=False, off_nsp=False):
	
	if off_nsp:
		print('[INFO] OFF NSP')
		encoder, inp_placeholder = get_astromer_encoder(pretrain_weights, version='second',trainable=trainable)
		embedding = encoder(inp_placeholder, training=trainable)
		mask = 1.-inp_placeholder['att_mask']
		x = embedding * mask
		x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)
		x = Dense(1024, activation='relu')(x)
		x = Dense(512, activation='relu')(x)
		x = Dense(256, activation='relu')(x)


	if clf_name == 'mlp_att_zero':
		encoder, inp_placeholder = get_astromer_encoder(pretrain_weights, version='zero', trainable=trainable)
		x = encoder(inp_placeholder, training=trainable)
		mask = 1.-inp_placeholder['mask_in']
		x = x * mask
		x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)
		x = Dense(1024, activation='relu')(x)
		x = Dense(512, activation='relu')(x)
		x = Dense(256, activation='relu')(x)

	# Multilayer Perceptron + Attention Embedding
	if clf_name == 'mlp_att' and not off_nsp:
		encoder, inp_placeholder = get_astromer_encoder(pretrain_weights, version='second',trainable=trainable)
		transform_layer = TransformLayer()
		embedding = encoder(inp_placeholder, training=trainable)
		_, x = transform_layer(embedding)
		mask = 1.-tf.slice(inp_placeholder['att_mask'], [0, 1, 0], [-1, -1, -1])
		x = x * mask
		x = tf.reduce_sum(x, 1)/tf.reduce_sum(mask, 1)
		x = Dense(1024, activation='relu')(x)
		x = Dense(512, activation='relu')(x)
		x = Dense(256, activation='relu')(x)

	# Just the [CLS] token
	if clf_name == 'mlp_cls' and not off_nsp:
		encoder, inp_placeholder = get_astromer_encoder(pretrain_weights, version='second',trainable=trainable)
		embedding = encoder(inp_placeholder, training=trainable)
		transform_layer = TransformLayer()
		transform_layer.trainable = trainable
		x, _ = transform_layer(embedding)

	# [CLS] token concatenated with the average of the observation tokens
	if clf_name == 'mlp_all' and not off_nsp:
		encoder, inp_placeholder = get_astromer_encoder(pretrain_weights, version='second',trainable=trainable)
		transform_layer = TransformLayer()
		embedding = encoder(inp_placeholder, training=trainable)
		x_cls, x_rec = transform_layer(embedding)
		mask = 1.-tf.slice(inp_placeholder['att_mask'], [0, 1, 0], [-1, -1, -1])
		x_rec = x_rec * mask
		x = tf.reduce_sum(x_rec, 1)/tf.reduce_sum(mask, 1)
		x_cls = tf.squeeze(x_cls, axis=1)
		x = tf.concat([x_cls, x], axis=1, name='concat_cls_reduced')

	# Output layer
	x = LayerNormalization(name='layer_norm')(x)
	y_pred = Dense(n_classes, name='output_layer')(x)
	y_pred = tf.reshape(y_pred, [-1, n_classes])
	return Model(inputs=inp_placeholder, outputs=y_pred, name=clf_name)


def run(opt):
	with open(os.path.join(opt.pre_weights, 'config.toml'), 'r') as file:
		config = toml.load(file)

	# ==========================================================================================
	train_batches, valid_batches, num_cls = load_classification_data(opt.data,
																	 window_size=config['ws'], 
																	 batch_size=opt.bs,
																	 version='second',
																	 off_nsp=config['off_nsp'])
	
	if opt.debug:
		train_batches = train_batches.take(1)
		valid_batches = valid_batches.take(1)
		opt.epochs = 10
		opt.bs = 16

	clf_model = create_classifier(opt.pre_weights, 
								  config['ws'], num_cls, 
								  clf_name=opt.clf_name, 
								  trainable=False,
								  off_nsp=config['off_nsp'])

	clf_model.compile(optimizer=Adam(opt.lr),
					  loss=CategoricalCrossentropy(from_logits=True),
					  metrics=['accuracy'])

	cbks = get_callbacks(opt.p)

	history = clf_model.fit(train_batches,
							epochs= opt.epochs,
							callbacks=cbks,
							validation_data=valid_batches)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--pre-weights', 
						default='./presentation/experiments/astromer_2/results/normal/pretraining', 
						type=str,
						help='Pretrain weights folder')
	parser.add_argument('--p', default='./presentation/experiments/astromer_2/results/test', type=str,
					help='Project folder. Here will be stored weights and logs')
	parser.add_argument('--data', default='./data/records/alcock/fold_0/alcock_100', type=str,
					help='Data folder where tf.record files are located')
	parser.add_argument('--clf-name', default='mlp_att', type=str,
						help='Classifier name: mlp_att, mlp_cls, mlp_all, mlp_att_zero')
	parser.add_argument('--gpu', default='-1', type=str,
						help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')


	parser.add_argument('--lr', default=1e-3, type=float,
						help='learning rate')
	parser.add_argument('--bs', default=64, type=int,
						help='Batch size')
	parser.add_argument('--patience', default=20, type=int,
						help='Earlystopping threshold in number of epochs')
	parser.add_argument('--epochs', default=100000, type=int,
						help='Number of epochs')
	parser.add_argument('--ws', default=200, type=int,
						help='windows size of the PSFs')
	parser.add_argument('--off-nsp', action='store_true', help='Input format without NSP (astromer-1)')


	opt = parser.parse_args()        
	run(opt)