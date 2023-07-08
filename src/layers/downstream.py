import tensorflow as tf
import argparse
import toml
import os

from src.models import get_ASTROMER_II
from src.data.preprocessing import standardize_batch, min_max_scaler

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

		astromer = get_ASTROMER_II(num_layers=config['layers'],
		                           num_heads=config['nh'],
		                           head_dim=config['hdim'],
		                           mixer_size=config['mixer'],
		                           dropout=config['dropout'],
		                           pe_base=1000,
		                           pe_dim=128,
		                           pe_c=1,
		                           window_size=config['ws'],
		                           encoder_mode=config['encoder_mode'])
		astromer.load_weights(os.path.join(self.pretrain_weights, 'weights')).expect_partial()
		astromer.trainable = False

		self.encoder = astromer.get_layer('encoder')
		self.trainable = trainable

	def call(self, inputs):
		values = standardize_batch(inputs['values'])
		times = min_max_scaler(inputs['times'])

		inp_shape = tf.shape(inputs['values'])
		cls_vector = tf.ones([inp_shape[0], 1, 1], dtype=tf.float32)
		values = tf.concat([cls_vector*-99., values], axis=1)

		times = tf.concat([1.-cls_vector, times], axis=1)
		mask = tf.concat([cls_vector, inputs['mask']], axis=1)


		encoder_input = {
			'magnitudes':values,
			'times':times,
			'att_mask': 1.-mask,
			'seg_emb': tf.zeros_like(mask)
		}

		x_emb = self.encoder(encoder_input, training=self.trainable)

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
		x_rec = tf.multiply(inputs, mask, name='rec_valid')  # 1: valid / 0: padding
		if self.reduce_to == 'sum':
			x_rec = tf.reduce_sum(x_rec, 1, name='rec_reduced')
			return x_rec
		if self.reduce_to == 'mean':
			x_rec = tf.reduce_sum(x_rec, 1, name='rec_reduced')
			n_valid = tf.reduce_sum(mask, axis=1, name='n_valid')
			return  tf.math.divide_no_nan(x_rec, n_valid,  name='avg_obs_tokens')
		return x_rec
