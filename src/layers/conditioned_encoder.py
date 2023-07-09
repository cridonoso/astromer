import tensorflow as tf

from src.layers.attention import HeadAttentionMulti
from src.layers.positional import positional_encoding, PositionalEncoder

from tensorflow.keras import Model


class MergingLayer(tf.keras.layers.Layer):
	def __init__(self, units, num_heads, head_dim, **kwargs):
		super(MergingLayer, self).__init__(**kwargs)
		d_model = num_heads*head_dim
		self.layer_0 = tf.keras.layers.Dense(units, activation='tanh')
		self.layer_1 = tf.keras.layers.Dense(d_model)
	def call(self, inputs):
		x = self.layer_0(inputs)
		return self.layer_1(x)

class AttentionBlock(tf.keras.layers.Layer):
	def __init__(self, head_dim, num_heads, mixer_size, dropout=0.1, **kwargs):
		super(AttentionBlock, self).__init__(**kwargs)
		self.head_dim = head_dim
		self.num_heads = num_heads
		self.mixer_size = mixer_size
		self.dropout = dropout

	def build(self, input_shape):
		self.mha = HeadAttentionMulti(self.head_dim, self.num_heads)
		self.ffn = MergingLayer(self.mixer_size, self.num_heads, self.head_dim, name='att_block_merging_layer')

		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = tf.keras.layers.Dropout(self.dropout)
		self.dropout2 = tf.keras.layers.Dropout(self.dropout)

	def call(self, x, training=False, mask=None, return_weights=False):
		attn_output, att_weights = self.mha(x, mask)  # (batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training=training)
		attn_output = self.layernorm1(attn_output)
		ffn_output  = self.ffn(attn_output)  # (batch_size, input_seq_len, d_model)
		ffn_output  = self.dropout2(ffn_output, training=training)
		ffn_output  = self.layernorm2(ffn_output)
		if return_weights:
			return ffn_output, att_weights
		return ffn_output

	def get_config(self):
		config = super().get_config()
		config.update({
			"head_dim": self.head_dim,
			"num_heads": self.num_heads,
			"mixer_size": self.mixer_size,
			"dropout": self.rate,
		})
		return config

class CondEncoder(tf.keras.layers.Layer):
	def __init__(self, 
				window_size,
				num_layers, 
				num_heads, 
				head_dim, 
				mixer_size=128,
				dropout=0.1, 
				pe_base=1000, 
				pe_dim=128,
				pe_c=1., 
				**kwargs):
		super(CondEncoder, self).__init__(**kwargs)

		self.window_size = window_size
		self.num_layers  = num_layers
		self.num_heads   = num_heads
		self.head_dim    = head_dim
		self.mixer_size  = mixer_size
		self.dropout     = dropout
		self.pe_base     = pe_base
		self.pe_c        = pe_c
		self.pe_dim      = pe_dim

		self.positional_encoder = PositionalEncoder(self.pe_dim, base=self.pe_base, c=self.pe_c, name='PosEncoding')

		self.att_time_block = AttentionBlock(self.head_dim, self.num_heads, self.mixer_size, dropout=self.dropout, name='time_att_block')
		self.att_mag_block  = AttentionBlock(self.head_dim, self.num_heads, self.mixer_size, dropout=self.dropout, name='mag_att_block')

		self.merge_att_layer =  MergingLayer(self.mixer_size, self.num_heads, self.head_dim, name='mag_time_att_merging_layer') 

		self.enc_layers = [AttentionBlock(self.head_dim, self.num_heads, self.mixer_size, dropout=self.dropout, name=f'att_block_{i}')
							for i in range(self.num_layers)]

	def call(self, data, training=False):
		# adding embedding and position encoding.
		x_pe     = self.positional_encoder(data['times'])
		time_att = self.att_time_block(x_pe)

		x_mag   = tf.concat([data['magnitudes'], data['seg_emb']], 2)
		mag_att = self.att_mag_block(x_mag)

		x  = tf.concat([time_att, mag_att], 2, name='concat_time_mag_att')

		layers_outputs = []
		for i in range(self.num_layers):
			z =  self.enc_layers[i](x, mask=data['att_mask'])
			layers_outputs.append(z)
		x = tf.reduce_mean(layers_outputs, 0)
		x = tf.reshape(x, [-1, self.window_size+1, self.num_heads*self.head_dim])
		return   x # (batch_size, input_seq_len, d_model)