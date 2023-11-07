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
				 data_name=None,
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
		self.data_name		 = data_name

		self.d_model = num_heads*head_dim

		self.inp_transform   = tf.keras.layers.Dense(self.d_model, name='inp_transform') # self.pe_dim
		
		if self.pe_type == 'APE':
			self.pe_transform = self.__select_pos_emb(pe_config)
		elif self.pe_type == 'RPE':
			self.rel_embedding   = tf.keras.layers.Dense(self.d_model, name='rel_embedding')
		elif self.pe_type == 'MixPE':
			self.pe_transform = self.__select_pos_emb(pe_config)
			self.rel_embedding   = tf.keras.layers.Dense(self.d_model, name='rel_embedding')
		elif self.pe_type == 'MixPE_v1':
			self.pe_transform = self.__select_pos_emb(pe_config)
			self.rel_embedding_q = tf.keras.layers.Dense(self.d_model, name='rel_embedding_q')
			self.rel_embedding_k = tf.keras.layers.Dense(self.d_model, name='rel_embedding_k')
		elif self.pe_type == 'ALiBi':
			context_position = tf.range(self.window_size)[:, None]
			memory_position = tf.range(self.window_size)[None, :]
			relative_position = tf.abs(memory_position - context_position)
			relative_position = tf.cast(relative_position, dtype=tf.float32)
			relative_position = tf.expand_dims(relative_position, axis=0)
		
			def get_slopes(n):
				def get_slopes_power_of_2(n):
					start = tf.constant((2**(-2**-(math.log2(n)-3))), dtype=tf.float32)
					ratio = start
					return [start * ratio**i for i in range(n)]

				if math.log2(n).is_integer():
					return get_slopes_power_of_2(n)
				else:
					closest_power_of_2 = 2**math.floor(math.log2(n))
					slopes = get_slopes_power_of_2(closest_power_of_2)
					slopes += get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
					return slopes

			slopes = tf.convert_to_tensor(get_slopes(self.num_heads), dtype=tf.float32) * -1
			self.alibi = tf.reshape(slopes, (1, self.num_heads, 1, 1)) * relative_position
			self.alibi = tf.reshape(self.alibi, (1, self.num_heads, self.window_size, self.window_size))

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
			'MixPE_v1': self.combine_type_MixPE_v1,
			'ALiBi': self.combine_type_ALiBi,
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
		pe_t = 0.

		if self.pe_func_name not in ['not_times', 'pea']:
			pe_t = self.pe_transform(inputs['times'])

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

	def combine_type_ALiBi(self, inputs, mask, training):
		''' Relative Positional Embedding '''

		emb_x = self.inp_transform(inputs['magnitudes'])
		emb_x = self.dropout_layer(emb_x, training=training)

		layers_outputs = []
		inputs_emb = [emb_x, self.alibi, layers_outputs]
		for i in range(self.num_layers):
			x, _ =  self.enc_layers[i](inputs_emb, mask=mask, training=training)
			layers_outputs.append(x)

		x = self.output_format(layers_outputs, self.window_size) 

		return x


	def combine_type_MixPE(self, inputs, mask, training):
		''' Absolute and Relative Positional Embedding '''

		emb_x = self.inp_transform(inputs['magnitudes']) 
		pe_t = 0.

		if self.pe_func_name not in ['not_times', 'pea']:
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

	def combine_type_MixPE_v1(self, inputs, mask, training):
		''' Absolute and Relative Positional Embedding '''

		emb_x = self.inp_transform(inputs['magnitudes']) 
		pe_t = 0.

		if self.pe_func_name not in ['not_times', 'pea']:
			pe_t = self.pe_transform(inputs['times'])

		pos_rel = inputs['times'] - tf.transpose(inputs['times'], perm=[0,2,1])
		Q_pe_rel_t = self.rel_embedding_q(pos_rel)
		K_pe_rel_t = self.rel_embedding_k(pos_rel)

		emb_x = self.dropout_layer(emb_x, training=training)
		pe_t = self.dropout_layer(pe_t, training=training)
		Q_pe_rel_t = self.dropout_layer(Q_pe_rel_t, training=training)
		K_pe_rel_t = self.dropout_layer(K_pe_rel_t, training=training)

		layers_outputs = []
		inputs_emb = [emb_x, pe_t, Q_pe_rel_t, K_pe_rel_t, layers_outputs, self.num_layers, 0]
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

		elif self.pe_func_name == 'not_times':
			return

		elif self.pe_func_name == 'pe':
			pe_transform = PositionalEmbedding(d_model		= self.d_model, 
											   base			= pe_config[self.pe_func_name]['base'], 
											   c			= pe_config[self.pe_func_name]['c'],  
											   use_mjd		= pe_config[self.pe_func_name]['use_mjd'], 
											   pe_trainable	= pe_config[self.pe_func_name]['pe_trainable'], 
											   initializer	= pe_config[self.pe_func_name]['initializer'], 
											   min_period	= pe_config[self.pe_func_name]['min_period'], 
											   max_period	= pe_config[self.pe_func_name]['max_period'],
											   data_name	= self.data_name,
											   name			= 'PosEncoding') # No puedo cambiar el nombre porque ya tengo el modelo entrenado
			

		elif self.pe_func_name == 'mlp':
			pe_transform = PosEmbeddingMLP(d_model	= self.d_model, 
										   m_layers = pe_config[self.pe_func_name]['m_layers'],
										   use_mjd  = pe_config[self.pe_func_name]['use_mjd'],
										   name		= 'pos_embedding_mlp') # No puedo cambiar el nombre porque ya tengo el modelo entrenado


		elif self.pe_func_name == 'rnn':
			pe_transform = PosEmbeddingRNN(d_model  = self.d_model, 
										   rnn_type = pe_config[self.pe_func_name]['rnn_type'], 
										   use_mjd  = pe_config[self.pe_func_name]['use_mjd'],
										   name		= 'rnn') 


		elif self.pe_func_name == 'tm':
			pe_transform = PosTimeModulation(d_model  	  = self.d_model, 
											 T_max		  = pe_config[self.pe_func_name]['T_max'], 
											 H			  = pe_config[self.pe_func_name]['H'],  
											 use_mjd	  = pe_config[self.pe_func_name]['use_mjd'],  
											 pe_trainable = pe_config[self.pe_func_name]['pe_trainable'],  
											 initializer  = pe_config[self.pe_func_name]['initializer'],
											 name		  = 'pos_time_modulation') # No puedo cambiar el nombre porque ya tengo el modelo entrenado
			

		elif self.pe_func_name == 'rnn_proj':
			pos_embbeding_rnn = PosEmbeddingRNN(d_model  = self.d_model, 
												rnn_type = pe_config[self.pe_func_name]['rnn_type'], 
												use_mjd  = pe_config[self.pe_func_name]['use_mjd'],
												name	 = 'pos_embbeding_rnn') 
			proj_w = tf.keras.layers.Dense(self.d_model, use_bias=True, name='proj_w')
			pe_transform = tf.keras.Sequential([pos_embbeding_rnn, proj_w], name='rnn_proj')


		elif self.pe_func_name == 'pe_mlp':
			pe_transform_a = PositionalEmbedding(d_model	 = self.d_model, 
												base		 = pe_config[self.pe_func_name]['base'], 
												c			 = pe_config[self.pe_func_name]['c'],  
												use_mjd		 = pe_config[self.pe_func_name]['use_mjd'], 
												pe_trainable = pe_config[self.pe_func_name]['pe_trainable'], 
												initializer	 = pe_config[self.pe_func_name]['initializer'], 
												min_period	 = pe_config[self.pe_func_name]['min_period'], 
												max_period	 = pe_config[self.pe_func_name]['max_period'],
												data_name	 = self.data_name,
												name		 = 'PosEncoding')
			pe_transform_b = PosEmbeddingMLP(d_model	= self.d_model, 
											 m_layers 	= pe_config[self.pe_func_name]['m_layers'],
											 use_mjd  	= pe_config[self.pe_func_name]['use_mjd'],
											 name		= 'pos_embedding_mlp')
			pe_transform = tf.keras.Sequential([pe_transform_a, pe_transform_b], name='pe_mlp')


		elif self.pe_func_name == 'pe_rnn_proj':
			pe_transform_a = PositionalEmbedding(d_model	  = self.d_model, 
												 base		  = pe_config[self.pe_func_name]['base'], 
												 c			  = pe_config[self.pe_func_name]['c'],  
												 use_mjd	  = pe_config[self.pe_func_name]['use_mjd'], 
												 pe_trainable = pe_config[self.pe_func_name]['pe_trainable'], 
												 initializer  = pe_config[self.pe_func_name]['initializer'], 
												 min_period	  = pe_config[self.pe_func_name]['min_period'], 
												 max_period	  = pe_config[self.pe_func_name]['max_period'],
												 data_name	  = self.data_name, 
												 name		  = 'PosEncoding') 
			pe_transform_b = PosEmbeddingRNN(d_model  = self.d_model, 
											 rnn_type = pe_config[self.pe_func_name]['rnn_type'], 
											 use_mjd  = pe_config[self.pe_func_name]['use_mjd'], 
											 name	  = 'pos_embbeding_rnn') 
			proj_w = tf.keras.layers.Dense(self.d_model, use_bias=True, name='proj_w')
			pe_transform = tf.keras.Sequential([pe_transform_a, pe_transform_b, proj_w], name='pe_rnn_proj')
			

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