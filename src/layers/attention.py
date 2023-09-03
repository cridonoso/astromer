import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask=None, return_mask=False):
	"""Calculate the attention weights.
	q, k, v must have matching leading dimensions.
	k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
	The mask has different shapes depending on its type(padding or look ahead)
	but it must be broadcastable for addition.

	Args:
	q: query shape == (..., seq_len_q, depth)
	k: key shape == (..., seq_len_k, depth)
	v: value shape == (..., seq_len_v, depth_v)
	mask: Float tensor with shape broadcastable
		  to (..., seq_len_q, seq_len_k). Defaults to None.

	Returns:
	output, attention_weights
	"""

	matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

	# scale matmul_qk
	dk = tf.cast(tf.shape(k)[-1], tf.float32)
	scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

	if mask is not None:
		steps = tf.shape(scaled_attention_logits)[2]
		mask = tf.tile(mask, [1,1,steps])
		mask_rshp_v = tf.reverse(mask, axis=[1])
		mask_rshp_h = tf.transpose(mask, [0,2,1])
		mask_rshp   = mask_rshp_v + mask_rshp_h
		mask_rshp   = tf.minimum(1., mask_rshp)
		mask_rshp   = tf.expand_dims(mask_rshp, 1)
		scaled_attention_logits += (mask_rshp*-1e9)

	# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
	attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
	output = tf.matmul(attention_weights, v, name='Z')  # (..., seq_len_q, depth_v)
	if return_mask:
		return output, attention_weights, mask_rshp
	return output, attention_weights

class HeadAttentionMulti(tf.keras.layers.Layer):
	def __init__(self, head_dim, num_heads):
		super(HeadAttentionMulti, self).__init__()
		self.num_heads = num_heads
		self.head_dim = head_dim
		
		self.d_model = self.num_heads * self.head_dim
		self.depth = self.d_model // self.num_heads # final dimension
		
		self.wq = tf.keras.layers.Dense(self.d_model, name='WQ')
		self.wk = tf.keras.layers.Dense(self.d_model, name='WK')
		self.wv = tf.keras.layers.Dense(self.d_model, name='WV')
		self.dense = tf.keras.layers.Dense(self.d_model, name='attmerge')

	def split_heads(self, x, batch_size, name='qkv'):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

	def call(self, x, training, mask=None):
		batch_size = tf.shape(x)[0]

		q = self.wq(x)  # (batch_size, seq_len, d_model)
		k = self.wk(x)  # (batch_size, seq_len, d_model)
		v = self.wv(x)  # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size, name='Q')  # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size, name='K')  # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size, name='V')  # (batch_size, num_heads, seq_len_v, depth)

		# scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
		# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
		scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask=mask)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention,
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
		
		return output, attention_weights

	def get_config(self):
		config = super().get_config()
		config.update({
			"head_dim": self.head_dim,
			"num_heads": self.num_heads,
		})
		return config

	def get_config(self):
		base_config = super().get_config()
		config = {
			"head_dim": self.head_dim,
			"num_heads": self.num_heads,
		}
		return {**base_config, **config}