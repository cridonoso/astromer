import multiprocessing as mp
import tensorflow as tf
import pandas as pd
import argparse
import toml
import os

from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.optimizers import Adam
from src.data.preprocessing import standardize_batch, min_max_scaler
from src.losses import custom_rmse, custom_bce
from src.metrics import custom_r2, custom_acc
from src.models import get_ASTROMER_II, get_ASTROMER
from src.data.masking import get_probed
from src.data.nsp import randomize
from src.data.loaders import format_input, load_data
from src.data.zero import format_inp_astromer, to_windows, standardize as standardize_a1, mask_dataset

from src.models.zero import get_ASTROMER, build_input
from src.data.zero import pretraining_pipeline

from src.models.second import build_input as build_input_II

def get_astromer_encoder(path, version='zero'):

	with open(os.path.join(path, 'config.toml'), 'r') as file:
		config = toml.load(file)

	if version == 'zero':	
		inp_placeholder = build_input(config['astromer']['window_size'])

		d_model = config['astromer']['heads'] * config['astromer']['head_dim']
		astromer = get_ASTROMER(num_layers=config['astromer']['layers'],
								d_model=d_model,
								num_heads=config['astromer']['heads'],
								dff=config['astromer']['dff'],
								base=config['positional']['base'],
								dropout=config['astromer']['dropout'],
								maxlen=config['astromer']['window_size'],
								pe_c=config['positional']['alpha'],
								no_train=False)

	if version == 'second':
		if 'nsp_normal_bigpe' in path:
			pe_dim = 256
		else:
			pe_dim = 128

		inp_placeholder = build_input_II(config['ws'])
		astromer = get_ASTROMER_II(num_layers=config['layers'],
								   num_heads=config['nh'],
								   head_dim=config['hdim'],
								   mixer_size=config['mixer'],
								   dropout=config['dropout'],
								   pe_base=1000,
								   pe_dim=pe_dim,
								   pe_c=1,
								   window_size=config['ws'],
								   encoder_mode=config['encoder_mode'])

	astromer.load_weights(os.path.join(path, 'weights')).expect_partial()
	encoder = astromer.get_layer('encoder')
	encoder.trainable = False
	return encoder, inp_placeholder

def load_classification_data(path, window_size, batch_size, version='zero', test=False):
	num_cls = pd.read_csv(os.path.join(path, 'objects.csv')).shape[0]

	if version == 'zero':
		train_batches = pretraining_pipeline(
				os.path.join(path, 'train'),
				batch_size,
				window_size,
				0.,
				0.,
				0.,
				False,
				True, #shuffle
				repeat=1,
				num_cls=num_cls,
				normalize='zero-mean',
				cache=True)
		valid_batches = pretraining_pipeline(
				os.path.join(path, 'val'),
				batch_size,
				window_size,
				0.,
				0.,
				0.,
				False,
				False, #shuffle
				repeat=1,
				num_cls=num_cls,
				normalize='zero-mean',
				cache=True)
		if test:
			test_batches = pretraining_pipeline(
			os.path.join(path, 'test'),
			batch_size,
			window_size,
			0.,
			0.,
			0.,
			False,
			False, #shuffle
			repeat=1,
			num_cls=num_cls,
			normalize='zero-mean',
			cache=True)
			return train_batches, valid_batches, test_batches, num_cls
		return train_batches, valid_batches, num_cls


	if version == 'second':
		train_batches = load_data(dataset=os.path.join(path, 'train'), 
								  batch_size=batch_size, 
								  probed=1.,  
								  window_size=window_size, 
								  nsp_prob=0., 
								  repeat=1, 
								  sampling=False,
								  shuffle=True,
								  num_cls=num_cls)
		valid_batches = load_data(dataset=os.path.join(path, 'val'), 
								  batch_size=batch_size, 
								  probed=1.,  
								  window_size=window_size, 
								  nsp_prob=0., 
								  repeat=1, 
								  sampling=False,
								  shuffle=False,
								  num_cls=num_cls)
		if test:
			test_batches = load_data(dataset=os.path.join(path, 'test'), 
						  batch_size=batch_size, 
						  probed=1.,  
						  window_size=window_size, 
						  nsp_prob=0., 
						  repeat=1, 
						  sampling=False,
						  shuffle=False,
						  num_cls=num_cls)
			return train_batches, valid_batches, test_batches, num_cls 
		return train_batches, valid_batches, num_cls


class AstromerEmbedding(tf.keras.layers.Layer):
	'''
	ASTROMER EMBEDDING LAYER 
	It loads pretrained weights on Astromer model and format input data to capture embeddings.
	Can be used in tensorflow models as a tensorflow layer
	'''
	def __init__(self, pretrain_weights, trainable=False, **kwargs):
		super(AstromerEmbedding, self).__init__(**kwargs)
		self.pretrain_weights = pretrain_weights
		self.kernel = '2'

		with open(os.path.join(self.pretrain_weights, 'config.toml'), 'r') as file:
			config = toml.load(file)

		try:
			self.astromer = get_ASTROMER_II(num_layers=config['layers'],
									   num_heads=config['nh'],
									   head_dim=config['hdim'],
									   mixer_size=config['mixer'],
									   dropout=config['dropout'],
									   pe_base=1000,
									   pe_dim=128,
									   pe_c=1,
									   window_size=config['ws'],
									   encoder_mode=config['encoder_mode'])
		except:
			print('[INFO] Astromer 2 arch did not match config. Trying with Astromer 1')
			d_model = config['astromer']['heads'] * config['astromer']['head_dim']
			self.astromer = get_ASTROMER(num_layers=config['astromer']['layers'],
										d_model=d_model,
										num_heads=config['astromer']['heads'],
										dff=config['astromer']['dff'],
										base=config['positional']['base'],
										dropout=config['astromer']['dropout'],
										maxlen=config['astromer']['window_size'],
										pe_c=config['positional']['alpha'],
										no_train=False)
			self.kernel = '1'

		self.astromer.load_weights(os.path.join(self.pretrain_weights, 'weights')).expect_partial()
		self.astromer.trainable = False
		self.config = config
		self.encoder = self.astromer.get_layer('encoder')
		self.trainable = trainable

	@tf.function
	def format_input(self, inputs, y=None):
		if self.kernel == '2':
			input_dict = get_probed(inputs, probed=1., njobs=mp.cpu_count())
			input_dict = randomize(inputs, 0.)
			input_dict = format_input(input_dict, cls_token=-99.)
			return input_dict

		if self.kernel == '1':
			input_dict = {}

			def partial(x):
				x = standardize_a1(x)
				return x['input']

			normed_input = tf.map_fn(lambda x: partial(x), inputs, 
									 fn_output_signature=tf.float32)
			
			input_dict['input'] = tf.slice(normed_input, [0, 0, 1], [-1, -1, 1])
			input_dict['times'] = tf.slice(normed_input, [0, 0, 0], [-1, -1, 1])
			input_dict['mask_out'] = tf.expand_dims(inputs['mask'], -1)
			input_dict['mask_in'] = 1. - input_dict['mask_out'] 
			return input_dict

	def evaluate_on_dataset(self, dataset):
		dataset = dataset.map(self.format_input)
		if self.kernel == '1':
			optimizer = Adam(self.config['lr'])
			self.astromer.compile(optimizer=optimizer)

		if self.kernel == '2':
			optimizer = AdamW(self.config['lr'])
			bce_factor    = 1.- self.config['rmse_factor']
			self.astromer.compile(rmse_factor=self.config['rmse_factor'], 
								  bce_factor=bce_factor, 
								  optimizer=optimizer)	

		self.astromer.evaluate(dataset)

	@tf.function
	def call(self, inputs):
		if self.kernel == '1':
			inputs = self.format_input(inputs)
			embedding = self.encoder(inputs, training=self.trainable)
			return None, embedding

		if self.kernel == '2':
			inputs, _ = self.format_input(inputs)
			x_emb = self.encoder(inputs, training=self.trainable)
			cls_token = tf.slice(x_emb, [0, 0, 0], [-1, 1, -1], name='nsp_tokens')
			rec_token = tf.slice(x_emb, [0, 1, 0], [-1, -1, -1], name='reconstruction_tokens')
			return tf.squeeze(cls_token, axis=1), rec_token	


class ReduceAttention(tf.keras.layers.Layer):
	'''
	Reduce embedding vector
	'''
	def __init__(self, reduce_to='mean', **kwargs):
		super(ReduceAttention, self).__init__(**kwargs)
		self.reduce_to = reduce_to

	def call(self, inputs, mask):
		mask = tf.expand_dims(mask, -1)
		x_rec = tf.multiply(inputs, mask, name='rec_valid')  # 1: valid / 0: padding

		if self.reduce_to == 'sum':
			x_rec = tf.reduce_sum(x_rec, 1, name='rec_reduced')
			return x_rec
		if self.reduce_to == 'mean':
			x_rec = tf.reduce_sum(x_rec, 1, name='rec_reduced')
			n_valid = tf.reduce_sum(mask, axis=1, name='n_valid')
			x_rec = tf.math.divide_no_nan(x_rec, n_valid,  name='avg_obs_tokens')
			return x_rec

		return x_rec
