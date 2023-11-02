import tensorflow as tf
import numpy as np

import math

from tensorflow.keras.layers import Layer

@tf.function
def get_positions(inputs):
	indices = tf.range(tf.shape(inputs)[1], dtype=tf.float32)
	indices = tf.expand_dims(indices, 0)
	indices = tf.tile(indices, [tf.shape(inputs)[0], 1])
	inputs = tf.expand_dims(indices, 2)
	return inputs


class PositionalEmbedding(tf.keras.layers.Layer):
	def __init__(self, d_model, base=1000, c=1, use_mjd=True, pe_trainable=False, 
			  	 initializer='pe_astromer', min_period=0.01, max_period=5.*365, data_name=None, **kwargs):
		super(PositionalEmbedding, self).__init__(**kwargs)
		self.d_model = d_model
		self.base = base
		self.c = tf.cast(c, tf.float32)
		self.min_period = min_period
		self.max_period = max_period
		self.data_name = data_name

		self.use_mjd = use_mjd
	
		self.pe_trainable = pe_trainable
		self.initializer = initializer
		self.init_freq = self.__select_initializer(initializer)

	def build(self, input_shape):
		self.w = self.add_weight(shape=(input_shape[-1], self.d_model),
								 initializer=self.init_freq,
								 trainable=self.pe_trainable,
								 name='pos_emb')

	def call(self, inputs):
        # Irregular or regular times
		if not self.use_mjd:
			inputs = get_positions(inputs)

		x_pe_embedding = tf.matmul(inputs, self.w)
		x_pe_embedding = self.get_concat(x_pe_embedding)
		#x_pe_embedding = tf.cast(x_pe_embedding, dtype=tf.float32)

		return x_pe_embedding

	
	def get_concat(self, x_pe_embedding):
		shape = tf.shape(x_pe_embedding)
		batch_size, seq_len, feature_dim = shape[0], shape[1], shape[2]

		#half_feature_dim = feature_dim // 2
#
		#sin = tf.math.sin(x_pe_embedding[:, :, :half_feature_dim])
		#cos = tf.math.cos(x_pe_embedding[:, :, half_feature_dim:])
		#
		#stack_sin_cos = tf.stack([sin, cos], axis=-1)       
		#astromer_concat = tf.reshape(stack_sin_cos, [batch_size, seq_len, -1])

		sin = tf.math.sin(x_pe_embedding[:, :, 0::2]) 
		cos = tf.math.cos(x_pe_embedding[:, :, 1::2])

		stack_sin_cos = tf.stack([sin, cos], axis=-1)          
		astromer_concat = tf.reshape(stack_sin_cos, [batch_size, seq_len, -1])
		return astromer_concat

	def __select_initializer(self, initializer):
		if initializer == 'pe_astromer':
			initializer = tf.constant_initializer(self.__get_frequency_ASTROMER().numpy()) 
		elif initializer == 'pe_vaswani':
			initializer = tf.constant_initializer(self.__get_frequency_Vaswani().numpy()) 
		elif initializer == 'pe_geom_prog':
			initializer = tf.constant_initializer(self.__get_frequency_GeomProg().numpy()) 
		else:
			print('You are using {} initializer.'.format(initializer))
		return initializer

	def __get_frequency_ASTROMER(self):
		dim_indices = tf.range(self.d_model, dtype=tf.float32)

		exponent = tf.divide(tf.multiply(self.c, dim_indices),
							tf.cast(self.d_model, tf.float32))

		frequency = tf.pow(tf.cast(self.base, dtype=tf.float32), exponent)
		frequency = tf.math.reciprocal(frequency)
		
		if self.data_name is not None:
			print('We are using the correction from {} data'.format(self.data_name))
			data_mean_ratio = {
				'alcock': 0.752,
				'atlas': 1.005,
				'ogle': 0.800,
				'kepler': 122.505,
				'kepler_alcock_linear': 0.739,
				'kepler_atlas_linear': 0.663,
				'kepler_ogle_linear': 0.829,
			}

			frequency = frequency * data_mean_ratio[self.data_name]

		return frequency

	def __get_frequency_Vaswani(self):
		dim_indices = tf.range(self.d_model//2, dtype=tf.float32)
		dim_indices = tf.repeat(dim_indices, 2)
		exponent = tf.divide(tf.multiply(2., dim_indices),
							 tf.cast(self.d_model, tf.float32))

		frequency = tf.pow(tf.cast(self.base, dtype=tf.float32), exponent)
		frequency = tf.math.reciprocal(frequency)
		return frequency

	def __get_frequency_GeomProg(self):
		t_init = tf.math.log(self.min_period)
		t_end = tf.math.log(self.max_period) 
		exponents = tf.linspace(t_init, t_end, self.d_model)
		angle_rates = tf.math.divide_no_nan(1.0, tf.exp(exponents))
		frequency = 2*math.pi*angle_rates
		return frequency

	def get_config(self):
		config = super().get_config()
		config.update({
			"d_model": self.d_model,
			"base": self.base,
			"c": self.c,
			"use_mjd": self.use_mjd,
			"pe_trainable": self.pe_trainable,
			"initializer": self.initializer,
			"min_period": self.min_period,
			"max_period": self.max_period,
			"data_name": self.data_name
		})
		return config


class PosEmbeddingMLP(tf.keras.layers.Layer):
	def __init__(self, d_model, m_layers=128, use_mjd=True, **kwargs):
		super(PosEmbeddingMLP, self).__init__(**kwargs)

		self.use_mjd = use_mjd
		self.d_model = d_model
		self.m_layers = m_layers

		self.mlp = tf.keras.Sequential([
							tf.keras.layers.Dense(self.m_layers, activation='gelu'),
							tf.keras.layers.Dense(self.d_model),
							], name='pos_mlp_emb')

	def call(self, inputs):
        # Irregular or regular times
		if not self.use_mjd:
			inputs = get_positions(inputs)

		x_pe_embedding = self.mlp(inputs)
		return x_pe_embedding

	def get_config(self):
		config = super().get_config()
		config.update({
			"d_model": self.d_model,
			"m_layers": self.m_layers,
			"use_mjd": self.use_mjd,
		})
		return config


class PosEmbeddingRNN(tf.keras.layers.Layer):
	def __init__(self, d_model, rnn_type='gru', use_mjd=True, **kwargs):
		super(PosEmbeddingRNN, self).__init__(**kwargs)

		self.use_mjd = use_mjd
		self.d_model = d_model
		self.rnn_type = rnn_type

		RNNLayer = self.__select_rnn_type(rnn_type)
		self.RNN = RNNLayer(self.d_model, return_sequences=True, name='pos_rnn_emb')

	def call(self, inputs):
        # Irregular or regular times
		if not self.use_mjd:
			inputs = get_positions(inputs)

		x_pe_embedding = self.RNN(inputs)
		return x_pe_embedding

	def __select_rnn_type(self, rnn_type):
		if rnn_type.lower() == 'rnn':
			RNNLayer = tf.keras.layers.SimpleRNN
		elif rnn_type.lower() == 'lstm':
			RNNLayer = tf.keras.layers.LSTM
		elif rnn_type.lower() == 'gru':
			RNNLayer = tf.keras.layers.GRU
		else:
			raise ValueError("Invalid RNN type. Choose from 'rnn', 'lstm' and 'gru'.")	
		return RNNLayer

	def get_config(self):
		config = super().get_config()
		config.update({
			"d_model": self.d_model,
			"rnn_type": self.m_layers,
			"use_mjd": self.use_mjd,
		})
		return config
	

class PosTimeModulation(tf.keras.layers.Layer):
	def __init__(self, d_model, T_max=1500, H=64, use_mjd=True, pe_trainable=True, 
			  	 initializer='random_normal', **kwargs):
		super(PosTimeModulation, self).__init__(**kwargs)

		self.use_mjd = use_mjd
		self.pe_trainable = pe_trainable
		self.initializer = initializer

		self.d_model = d_model
		self.T_max = T_max
		self.H = H
		self.ar = tf.range(self.H, dtype='float32')[tf.newaxis, tf.newaxis, :]

		self.alpha_sin = self.add_weight(shape=(self.H, self.d_model),
										  initializer=initializer,
									 	  trainable=pe_trainable,
									 	  name='alpha_sin')
		self.alpha_cos = self.add_weight(shape=(self.H, self.d_model),
										  initializer=initializer,
									 	  trainable=pe_trainable,
									 	  name='alpha_cos')
		self.beta_sin = self.add_weight(shape=(self.H, self.d_model),
										  initializer=initializer,
									 	  trainable=pe_trainable,
									 	  name='beta_sin')
		self.beta_cos = self.add_weight(shape=(self.H, self.d_model),
										  initializer=initializer,
									 	  trainable=pe_trainable,
									 	  name='beta_cos')

	def call(self, inputs):
        # Irregular or regular times
		if not self.use_mjd:
			inputs = get_positions(inputs)

		time_emb_sin = tf.math.sin((2 * math.pi * self.ar * tf.repeat(inputs, repeats=self.H, axis=-1)) / self.T_max)
		time_emb_cos = tf.math.cos((2 * math.pi * self.ar * tf.repeat(inputs, repeats=self.H, axis=-1)) / self.T_max)
		
		alpha = tf.matmul(time_emb_sin, self.alpha_sin) + tf.matmul(time_emb_cos, self.alpha_cos)
		beta = tf.matmul(time_emb_sin, self.beta_sin) + tf.matmul(time_emb_cos, self.beta_cos)

		return alpha, beta

	def get_config(self):
		config = super().get_config()
		config.update({
			"d_model": self.d_model,
			"T_max": self.T_max,
			"H": self.H,
			"use_mjd": self.use_mjd,
			"pe_trainable": self.pe_trainable,
			"initializer": self.initializer
		})
		return config


class PosRelativeCont(tf.keras.layers.Layer):
	def __init__(self, d_model, use_mjd=True, **kwargs):
		super(PosRelativeCont, self).__init__(**kwargs)

		self.use_mjd = use_mjd
		self.d_model = d_model

		self.pe_transform = tf.keras.layers.Dense(self.d_model, name='pe_transform')

	def call(self, inputs):
        # Irregular or regular times ( Tiene sentido hacerlo aca? )
		if not self.use_mjd:
			inputs = get_positions(inputs)

		pos_rel = inputs - tf.transpose(inputs, perm=[0,2,1])
		x_pos_rel_embedding = self.pe_transform(pos_rel)

		return x_pos_rel_embedding

	def get_config(self):
		config = super().get_config()
		config.update({
			"d_model": self.d_model,
			"use_mjd": self.use_mjd,
		})
		return config