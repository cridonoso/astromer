import tensorflow as tf

from src.layers.attblock import AttentionBlock
from src.layers.positional import *
from src.layers.nsp import ClassToken
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras import Model

class Encoder(tf.keras.Model):
	""" Encoder as it was defined in Astromer I """
	def __init__(self, 
				 window_size,
				 num_layers, 
				 num_heads, 
				 head_dim, 
				 mixer_size=128,
				 dropout=0.1, 
				 pe_type='APE', # Para despues se deberia llamar encoder_type
				 pe_config=None,
				 pe_func_name='pe',
				 residual_type=None,
				 average_layers=False,
				 **kwargs):
		super(Encoder, self).__init__(**kwargs)

		self.window_size 	 = window_size
		self.num_layers  	 = num_layers
		self.num_heads   	 = num_heads
		self.head_dim    	 = head_dim
		self.mixer_size  	 = mixer_size
		self.dropout     	 = dropout
		self.pe_type		 = pe_type
		self.residual_type	 = residual_type
		self.pe_func_name 	 = pe_func_name
		self.average_layers  = average_layers

		self.d_model = num_heads*head_dim

		self.inp_transform   = tf.keras.layers.Dense(self.d_model, name='inp_transform') # self.pe_dim
		
		if self.pe_type == 'APE':
			self.pe_transform = self.__select_pos_emb(pe_config)
		elif self.pe_type == 'RPE':
			self.rel_embedding   = tf.keras.layers.Dense(self.d_model, name='rel_embedding')
		elif self.pe_type == 'MixPE':
			self.pe_transform = self.__select_pos_emb(pe_config)
			self.rel_embedding   = tf.keras.layers.Dense(self.d_model, name='rel_embedding')
		else:
			raise ValueError("Unknown positional encoding type")


		self.enc_layers = [AttentionBlock(self.head_dim, self.num_heads, self.mixer_size, dropout=self.dropout, 
										  pe_type=self.pe_type, pe_func_name=self.pe_func_name, residual_type=self.residual_type, 
										  name=f'att_layer_{i}')
							for i in range(self.num_layers)]


		self.combiner_layers = {
			'APE': self.combine_type_APE,
			'RPE': self.combine_type_RPE,
			'MixPE': self.combine_type_MixPE,
		}
					
		self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

	def input_format(self, inputs):
		if 'seg_emb' in inputs.keys():
			window_size = self.window_size + 1 # if seg_emb exists then NSP is being applied
			x = tf.concat([inputs['magnitudes'], inputs['seg_emb']], axis=2, name='concat_mag_segemb')
		else:
			window_size = self.window_size
			x = inputs['magnitudes']

		x_transformed = self.inp_transform(x)   
		x_pe = self.positional_encoder(inputs['times'])
		x = x_transformed + x_pe   
		return x , window_size

	def output_format(self, outputs, window_size):
		if self.average_layers:
			x = tf.reduce_mean(outputs, 0)
		else:
			x = outputs[-1]

		x = tf.reshape(x, [-1, window_size, self.num_heads*self.head_dim])
		return x

	def call(self, inputs, training=False):
		x = self.combiner_layers[self.pe_type](inputs, 
											   mask=inputs['att_mask'], 
											   training=training)

		return  x # (batch_size, input_seq_len, d_model)
				

	def combine_type_APE(self, inputs, mask, training):
		''' Absolute Positional Embedding '''

		emb_x = self.inp_transform(inputs['magnitudes']) 
		pe_t = 0

		if self.pe_func_name not in ['not_pe_module', 'pea']:
			pe_t = self.pe_transform(inputs['times'])

		#emb_x = self.dropout_layer(emb_x, training=training)
		#pe_t = self.dropout_layer(pe_t, training=training)

		layers_outputs = []
		inputs_emb = [emb_x, pe_t, self.dropout_layer, layers_outputs, self.num_layers, 0]
		for i in range(self.num_layers):
			inputs_emb[-1] = i
			x, _ =  self.enc_layers[i](inputs_emb, mask=mask, training=training)
			layers_outputs.append(x)

		x = self.output_format(layers_outputs, self.window_size) 

		if self.pe_func_name == 'pea':
			pe_t = self.pe_transform(inputs['times'])
			x = x + pe_t

		return x


	def combine_type_RPE(self, inputs, mask, training):
		''' Relative Positional Embedding '''

		emb_x = self.inp_transform(inputs['magnitudes']) 

		pos_rel = inputs['times'] - tf.transpose(inputs['times'], perm=[0,2,1])
		pe_rel_t = self.rel_embedding(pos_rel)

		emb_x = self.dropout_layer(emb_x, training=training)
		pe_rel_t = self.dropout_layer(pe_rel_t, training=training)

		layers_outputs = []
		inputs_emb = [emb_x, pe_rel_t]
		for i in range(self.num_layers):
			x, _ =  self.enc_layers[i](inputs_emb, mask=mask, training=training)
			layers_outputs.append(x)

		x = self.output_format(layers_outputs, self.window_size) 

		return x


	def combine_type_MixPE(self, inputs, mask, training):
		''' Absolute and Relative Positional Embedding '''

		emb_x = self.inp_transform(inputs['magnitudes']) 
		pe_t = 0

		if self.pe_func_name not in ['not_pe_module', 'pea']:
			pe_t = self.pe_transform(inputs['times'])

		pos_rel = inputs['times'] - tf.transpose(inputs['times'], perm=[0,2,1])
		pe_rel_t = self.rel_embedding(pos_rel)

		emb_x = self.dropout_layer(emb_x, training=training)
		pe_t = self.dropout_layer(pe_t, training=training)
		pe_rel_t = self.dropout_layer(pe_rel_t, training=training)

		layers_outputs = []
		inputs_emb = [emb_x, pe_t, pe_rel_t, layers_outputs, self.num_layers, 0]
		for i in range(self.num_layers):
			inputs_emb[-1] = i
			x, _ =  self.enc_layers[i](inputs_emb, mask=mask, training=training)
			layers_outputs.append(x)

		x = self.output_format(layers_outputs, self.window_size) 

		if self.pe_func_name == 'pea':
			pe_t = self.pe_transform(inputs['times'])
			x = x + pe_t

		return x


	def __select_pos_emb(self, pe_config):
		if pe_config is None:
			raise Exception("You don't use correctly the config pe path.")

		if self.pe_func_name == 'not_pe':
			pass

		elif self.pe_func_name == 'pe':
			pe_transform = PositionalEmbedding(d_model		= self.d_model, 
											   base			= pe_config[self.pe_func_name]['base'], 
											   c			= pe_config[self.pe_func_name]['c'],  
											   use_mjd		= pe_config[self.pe_func_name]['use_mjd'], 
											   pe_trainable	= pe_config[self.pe_func_name]['pe_trainable'], 
											   initializer	= pe_config[self.pe_func_name]['initializer'], 
											   min_period	= pe_config[self.pe_func_name]['min_period'], 
											   max_period	= pe_config[self.pe_func_name]['max_period'],
											   name			= 'PosEncoding')
			
		elif self.pe_func_name == 'pe_mlp':
			pe_transform = PosEmbeddingMLP(d_model	= self.d_model, 
										   m_layers = pe_config[self.pe_func_name]['m_layers'],
										   use_mjd  = pe_config[self.pe_func_name]['use_mjd'],
										   name		= 'pos_embedding_mlp')

		elif self.pe_func_name == 'pe_rnn':
			pe_transform = PosEmbeddingRNN(d_model  = self.d_model, 
										   rnn_type = pe_config[self.pe_func_name]['rnn_type'], 
										   use_mjd  = pe_config[self.pe_func_name]['use_mjd'],
										   name		= 'pos_embedding_rnn') 

		elif self.pe_func_name == 'pe_tm':
			pe_transform = PosTimeModulation(d_model  	  = self.d_model, 
											 T_max		  = pe_config[self.pe_func_name]['T_max'], 
											 H			  = pe_config[self.pe_func_name]['H'],  
											 use_mjd	  = pe_config[self.pe_func_name]['use_mjd'],  
											 pe_trainable = pe_config[self.pe_func_name]['pe_trainable'],  
											 initializer  = pe_config[self.pe_func_name]['initializer'],
											 name		  = 'pos_time_modulation')

		elif self.pe_func_name == 'pe_astromer_mlp':
			pass

		else:
			raise ValueError("Unknown positional encoding name function")

		return pe_transform


class ConcatEncoder(Encoder):
	def input_format(self, inputs):
		x_pe = self.positional_encoder(inputs['times'])
		if 'seg_emb' in inputs.keys():
			window_size = self.window_size + 1
			x = tf.concat([x_pe, inputs['magnitudes'], inputs['seg_emb']], 2)
		else:
			window_size = self.window_size
			x = tf.concat([x_pe, inputs['magnitudes']], 2)
		return x, window_size

class NSPEncoder(Encoder):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

		self.cls_token = ClassToken()
		self.concat_cls    = Concatenate(axis=1, name='concat_cls')

	def input_format(self, inputs):
		x = tf.concat([inputs['magnitudes'], inputs['seg_emb']], axis=2, name='concat_mag_segemb')
		
		x_transformed = self.inp_transform(x)   
		x_pe = self.positional_encoder(inputs['times'])
		x = x_transformed + x_pe   

		x_cls = self.cls_token(x)
		x = self.concat_cls([x_cls, x])

		window_size = self.window_size + 1
		msk_cls_tkn = tf.zeros([tf.shape(x)[0], 1, 1])
		inputs['att_mask'] = self.concat_cls([msk_cls_tkn, inputs['att_mask']])

		return x, window_size