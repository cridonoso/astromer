import tensorflow as tf

from src.layers.attention import HeadAttentionMulti

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
		super(AttentionBlock, self).__init__(head_dim, num_heads, mixer_size, dropout=0.1, **kwargs)
		self.head_dim = head_dim
		self.num_heads = num_heads
		self.mixer_size = mixer_size
		self.dropout = dropout
		self.mha = HeadAttentionMulti(self.head_dim, self.num_heads)
		self.ffn = MergingLayer(self.mixer_size, self.num_heads, self.head_dim, name='att_block_merging_layer')
		self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout1 = tf.keras.layers.Dropout(self.dropout)
		self.dropout2 = tf.keras.layers.Dropout(self.dropout)

	def call(self, x, training, mask=None, return_weights=False):
		attn_output, att_weights = self.mha(x, mask)  # (batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training=training)
		attn_output = self.layernorm1(attn_output, training=training)
		ffn_output  = self.ffn(attn_output)  # (batch_size, input_seq_len, d_model)
		ffn_output  = self.dropout2(ffn_output, training=training)
		ffn_output  = self.layernorm2(ffn_output, training=training)
		if return_weights:
			return ffn_output, att_weights
		return ffn_output

class AttentionBlock_astrospec(AttentionBlock):
	def __init__(self, head_dim, num_heads, mixer_size, dropout=0.1, **kwargs):
		super(AttentionBlock_astrospec, self).__init__(head_dim, num_heads, mixer_size, dropout=0.1, **kwargs)
		self.skip_after_mha=tf.keras.layers.Dense(self.head_dim*self.num_heads)
		self.skip_after_ffn=tf.keras.layers.Dense(self.head_dim*self.num_heads)

	def call(self, x, training, mask=None, return_weights=False):
		attn_output, att_weights = self.mha(x, mask)  # (batch_size, input_seq_len, d_model)
		attn_output = self.dropout1(attn_output, training=training)
		residual_mask=tf.where(tf.equal(mask,1), tf.zeros_like(mask), tf.ones_like(mask))
		attn_output = self.layernorm1(self.skip_after_mha(x*residual_mask)+attn_output, training=training)
		ffn_output  = self.ffn(attn_output)  # (batch_size, input_seq_len, d_model)
		ffn_output  = self.dropout2(ffn_output, training=training)
		ffn_output  = self.layernorm2(self.skip_after_ffn(attn_output)+ffn_output, training=training)
		if return_weights:
			return ffn_output, att_weights
		return ffn_output

	def get_config(self):
		config = super().get_config()
		config.update({
			"head_dim": self.head_dim,
			"num_heads": self.num_heads,
			"dff": self.dff,
			"dropout": self.rate,
			"mha": serialize_keras_object(self.mha),
			"ffn": serialize_keras_object(self.ffn),
		})
		return config
	
	@classmethod
	def from_config(cls, config):
		mha_config = config.pop("mha")
		ffn_config = config.pop("ffn")
		mha_config = deserialize_keras_object(mha_config)
		ffn_config = deserialize_keras_object(ffn_config)
		return cls(mha_config, ffn_config, **config)