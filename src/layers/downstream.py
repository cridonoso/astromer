import multiprocessing as mp
import tensorflow as tf
import argparse
import toml
import os

from tensorflow.keras.optimizers.experimental import AdamW
from src.data.preprocessing import standardize_batch, min_max_scaler
from src.losses import custom_rmse, custom_bce
from src.metrics import custom_r2, custom_acc
from src.models import get_ASTROMER_II
from src.data.masking import get_probed
from src.data.nsp import randomize
from src.data.loaders import format_input

class AstromerEmbedding(tf.keras.layers.Layer):
	'''
	ASTROMER EMBEDDING LAYER 
	It loads pretrained weights on Astromer model and format input data to capture embeddings.
	Can be used in tensorflow models as a tensorflow layer
	'''
	def __init__(self, pretrain_weights, trainable=False, **kwargs):
		super(AstromerEmbedding, self).__init__(**kwargs)
		self.pretrain_weights = pretrain_weights

		with open(os.path.join(self.pretrain_weights, 'config.toml'), 'r') as file:
			config = toml.load(file)

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

		self.astromer.load_weights(os.path.join(self.pretrain_weights, 'weights')).expect_partial()
		self.astromer.trainable = False

		self.config = config
		self.encoder = self.astromer.get_layer('encoder')
		self.trainable = trainable

	def format_input(self, inputs, y=None):
		input_dict = get_probed(inputs, probed=1., njobs=mp.cpu_count())
		input_dict = randomize(inputs, 0.)
		input_dict = format_input(input_dict, cls_token=-99.)
		return input_dict

	def evaluate_on_dataset(self, dataset):
		dataset = dataset.map(self.format_input)
		optimizer = AdamW(self.config['lr'])
		bce_factor    = 1.- self.config['rmse_factor']
		self.astromer.compile(rmse_factor=self.config['rmse_factor'], bce_factor=bce_factor, optimizer=optimizer)
		self.astromer.evaluate(dataset)

	def call(self, inputs):
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
			return  tf.math.divide_no_nan(x_rec, n_valid,  name='avg_obs_tokens')
		return x_rec
