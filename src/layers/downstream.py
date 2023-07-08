import tensorflow as tf
import argparse
import toml
import os

from src.models import get_ASTROMER_II
from src.data.preprocessing import standardize_batch, min_max_scaler

class AstromerEmbedding(tf.keras.layers.Layer):
	def __init__(self, pretrain_weights, trainable=False):
		super(AstromerEmbedding, self).__init__()
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
		mask = tf.concat([cls_vector, tf.expand_dims(inputs['mask'], -1)], axis=1)


		encoder_input = {
			'magnitudes':values,
			'times':times,
			'att_mask': 1.-mask,
			'seg_emb': tf.zeros_like(mask)
		}

		x_emb = self.encoder(encoder_input, training=self.trainable)

		cls_token = tf.slice(x_emb, [0, 0, 0], [-1, 1, -1], name='nsp_tokens')
		rec_token = tf.slice(x_emb, [0, 1, 0], [-1, -1, -1], name='reconstruction_tokens')

		return cls_token, rec_token, encoder_input