import tensorflow as tf


class HeadAttentionMulti(tf.keras.layers.Layer):
	def __init__(self, head_dim, num_heads, pe_type,  pe_func_name, residual_type, **kwargs):
		super(HeadAttentionMulti, self).__init__(**kwargs)
		self.num_heads = num_heads
		self.head_dim = head_dim
		self.pe_type = pe_type
		self.pe_func_name = pe_func_name
		self.residual_type = residual_type
		
		self.d_model = self.num_heads * self.head_dim
		self.depth = self.d_model // self.num_heads # final dimension
		
		self.wq = tf.keras.layers.Dense(self.d_model, name='WQ')
		self.wk = tf.keras.layers.Dense(self.d_model, name='WK')
		self.wv = tf.keras.layers.Dense(self.d_model, name='WV')
		self.dense = tf.keras.layers.Dense(self.d_model, name='attmerge')

		self.combine_attention = {
			'APE': self.combine_attention_APE,
			'RPE': self.combine_attention_RPE,
			'MixPE': self.combine_attention_MixPE,
			'MixPE_v1': self.combine_attention_MixPE_v1,
			'ALiBi': self.combine_attention_ALiBi,
		}

	def split_heads(self, x, batch_size, name='qkv'):
		"""Split the last dimension into (num_heads, depth).
		Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
		"""
		x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
		return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

	def call(self, inputs, mask=None, training=False):
		output, attention_weights = self.combine_attention[self.pe_type](inputs, mask, training)
		return output, attention_weights

	def combine_attention_APE(self, inputs, mask, training):
		emb_x, pe_t, dropout, layers_outputs, num_layers, i_layer = inputs
		batch_size = tf.shape(emb_x)[0]

		x = self.get_input_embedding(emb_x, pe_t, dropout, layers_outputs, training)

		q = self.wq(x) # (batch_size, seq_len, d_model)
		k = self.wk(x) # (batch_size, seq_len, d_model)
		v = self.wv(x) # (batch_size, seq_len, d_model)

		q = self.split_heads(q, batch_size, name='Q') # (batch_size, num_heads, seq_len_q, depth)
		k = self.split_heads(k, batch_size, name='K') # (batch_size, num_heads, seq_len_k, depth)
		v = self.split_heads(v, batch_size, name='V') # (batch_size, num_heads, seq_len_v, depth)

		matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)

		# Residual connections
		if i_layer > 0:
			Wq_pos_Wk_pos = 0.
			if self.residual_type is not None:
				Wq_pos_Wk_pos = self.get_p2p_term(pe_t, i_layer, num_layers, Wq_pos_Wk_pos, batch_size)

			matmul_qk += Wq_pos_Wk_pos

		# scale matmul_qk
		dk = tf.cast(tf.shape(k)[-1], tf.float32)
		scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

		if mask is not None:
			print('Using masking...')
			scaled_attention_logits = self.get_mask(scaled_attention_logits, mask)

		# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
		attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
		scaled_attention = tf.matmul(attention_weights, v, name='Z')  # (..., seq_len_q, depth_v)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention,
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)

		return output, attention_weights
	

	def combine_attention_RPE(self, inputs, mask, training):
		emb_x, pe_rel_t = inputs
		batch_size = tf.shape(emb_x)[0]

		Q_c, Q_r = self.wq(emb_x), self.wq(pe_rel_t)
		K_c, K_r = self.wk(emb_x), self.wk(pe_rel_t)
		V_c = self.wv(emb_x)

		Q_c, Q_r = self.split_heads(Q_c, batch_size, name='Q_c'), self.split_heads(Q_r, batch_size, name='Q_r')  # (batch_size, num_heads, seq_len_q, depth)
		K_c, K_r = self.split_heads(K_c, batch_size, name='K_c'), self.split_heads(K_r, batch_size, name='K_r')  # (batch_size, num_heads, seq_len_k, depth)
		V_c = self.split_heads(V_c, batch_size, name='V_c')  # (batch_size, num_heads, seq_len_v, depth)

		# Scaled dot product
		matmul_Qc_Kc = tf.matmul(Q_c, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)
		matumul_Qr_Kc = tf.matmul(Q_r, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)
		matumul_Qc_Kr = tf.matmul(Q_c, K_r, transpose_b=True) # (..., seq_len_q, seq_len_k)

		matmul_qk = matmul_Qc_Kc + matumul_Qr_Kc + matumul_Qc_Kr

		# Residual connections
		if self.residual_type is not None:
			raise TypeError("RPE doesn't have residual connections")

		# scale matmul_qk
		dk = tf.cast(tf.shape(K_c)[-1], tf.float32)
		scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

		if mask is not None:
			print('Using masking...')
			scaled_attention_logits = self.get_mask(scaled_attention_logits, mask)

		# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
		attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
		scaled_attention = tf.matmul(attention_weights, V_c, name='Z')  # (..., seq_len_q, depth_v)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention,
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)

		return output, attention_weights


	def combine_attention_ALiBi(self, inputs, mask, training):
		emb_x, alibi, layers_outputs = inputs
		batch_size = tf.shape(emb_x)[0]

		if len(layers_outputs) > 0:	
			emb_x = layers_outputs[-1]

		Q_c = self.wq(emb_x)
		K_c = self.wk(emb_x)
		V_c = self.wv(emb_x)

		Q_c = self.split_heads(Q_c, batch_size, name='Q_c')  # (batch_size, num_heads, seq_len_q, depth)
		K_c = self.split_heads(K_c, batch_size, name='K_c')  # (batch_size, num_heads, seq_len_k, depth)
		V_c = self.split_heads(V_c, batch_size, name='V_c')  # (batch_size, num_heads, seq_len_v, depth)

		# Scaled dot product
		matmul_qk = tf.matmul(Q_c, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)
		matmul_qk += alibi

		# Residual connections
		if self.residual_type is not None:
			raise TypeError("RPE doesn't have residual connections")

		# scale matmul_qk
		dk = tf.cast(tf.shape(K_c)[-1], tf.float32)
		scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

		if mask is not None:
			print('Using masking...')
			scaled_attention_logits = self.get_mask(scaled_attention_logits, mask)

		# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
		attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
		scaled_attention = tf.matmul(attention_weights, V_c, name='Z')  # (..., seq_len_q, depth_v)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention,
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)

		return output, attention_weights

	def combine_attention_MixPE(self, inputs, mask, training):
		emb_x, pe_t, pe_rel_t, layers_outputs, num_layers, i_layer = inputs
		batch_size = tf.shape(emb_x)[0]

		if len(layers_outputs) == 0:	
			Q_c, Q_r = self.wq(emb_x), self.wq(pe_rel_t)
			K_c, K_r = self.wk(emb_x), self.wk(pe_rel_t)
			V_c = self.wv(emb_x)

			Q_c, Q_r = self.split_heads(Q_c, batch_size, name='Q_c'), self.split_heads(Q_r, batch_size, name='Q_r')  # (batch_size, num_heads, seq_len_q, depth)
			K_c, K_r = self.split_heads(K_c, batch_size, name='K_c'), self.split_heads(K_r, batch_size, name='K_r')  # (batch_size, num_heads, seq_len_k, depth)
			V_c = self.split_heads(V_c, batch_size, name='V_c')  # (batch_size, num_heads, seq_len_v, depth)

			# Scaled dot product
			matmul_Qc_Kc = tf.matmul(Q_c, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)
			matmul_Qr_Kc = tf.matmul(Q_r, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)
			matmul_Qc_Kr = tf.matmul(Q_c, K_r, transpose_b=True) # (..., seq_len_q, seq_len_k)

			matmul_qk = matmul_Qc_Kc + matmul_Qr_Kc + matmul_Qc_Kr

		else: 
			x = layers_outputs[-1]
			Q_c = self.wq(x) # (batch_size, seq_len, d_model)
			K_c = self.wk(x) # (batch_size, seq_len, d_model)
			V_c = self.wv(x) # (batch_size, seq_len, d_model)

			Q_c = self.split_heads(Q_c, batch_size, name='Q_c') # (batch_size, num_heads, seq_len_q, depth)
			K_c = self.split_heads(K_c, batch_size, name='K_c') # (batch_size, num_heads, seq_len_k, depth)
			V_c = self.split_heads(V_c, batch_size, name='V_c') # (batch_size, num_heads, seq_len_v, depth)

			matmul_qk = tf.matmul(Q_c, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)

		# Residual connections
		if i_layer > 0:
			Wq_pos_Wk_pos = 0.
			if self.residual_type is not None:
				Wq_pos_Wk_pos = self.get_p2p_term(pe_t, i_layer, num_layers, Wq_pos_Wk_pos, batch_size)

			matmul_qk += Wq_pos_Wk_pos

		# scale matmul_qk
		dk = tf.cast(tf.shape(K_c)[-1], tf.float32)
		scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

		if mask is not None:
			print('Using masking...')
			scaled_attention_logits = self.get_mask(scaled_attention_logits, mask)

		# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
		attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
		scaled_attention = tf.matmul(attention_weights, V_c, name='Z')  # (..., seq_len_q, depth_v)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention,
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)
		
		return output, attention_weights
	

	def combine_attention_MixPE_v1(self, inputs, mask, training):
		emb_x, pe_t, Q_pe_rel_t, K_pe_rel_t, layers_outputs, num_layers, i_layer = inputs
		batch_size = tf.shape(emb_x)[0]

		if len(layers_outputs) == 0:	
			Q_c = self.wq(emb_x)
			K_c = self.wk(emb_x)
			V_c = self.wv(emb_x)

			Q_c, Q_r = self.split_heads(Q_c, batch_size, name='Q_c'), self.split_heads(Q_pe_rel_t, batch_size, name='Q_r')  # (batch_size, num_heads, seq_len_q, depth)
			K_c, K_r = self.split_heads(K_c, batch_size, name='K_c'), self.split_heads(K_pe_rel_t, batch_size, name='K_r')  # (batch_size, num_heads, seq_len_k, depth)
			V_c = self.split_heads(V_c, batch_size, name='V_c')  # (batch_size, num_heads, seq_len_v, depth)

			# Scaled dot product
			matmul_Qc_Kc = tf.matmul(Q_c, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)
			matmul_Qr_Kc = tf.matmul(Q_r, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)
			matmul_Qc_Kr = tf.matmul(Q_c, K_r, transpose_b=True) # (..., seq_len_q, seq_len_k)

			matmul_qk = matmul_Qc_Kc + matmul_Qr_Kc + matmul_Qc_Kr

		else: 
			x = layers_outputs[-1]
			Q_c = self.wq(x) # (batch_size, seq_len, d_model)
			K_c = self.wk(x) # (batch_size, seq_len, d_model)
			V_c = self.wv(x) # (batch_size, seq_len, d_model)

			Q_c = self.split_heads(Q_c, batch_size, name='Q_c') # (batch_size, num_heads, seq_len_q, depth)
			K_c = self.split_heads(K_c, batch_size, name='K_c') # (batch_size, num_heads, seq_len_k, depth)
			V_c = self.split_heads(V_c, batch_size, name='V_c') # (batch_size, num_heads, seq_len_v, depth)

			matmul_qk = tf.matmul(Q_c, K_c, transpose_b=True) # (..., seq_len_q, seq_len_k)

		# Residual connections
		if i_layer > 0:
			Wq_pos_Wk_pos = 0.
			if self.residual_type is not None:
				Wq_pos_Wk_pos = self.get_p2p_term(pe_t, i_layer, num_layers, Wq_pos_Wk_pos, batch_size)

			matmul_qk += Wq_pos_Wk_pos

		# scale matmul_qk
		dk = tf.cast(tf.shape(K_c)[-1], tf.float32)
		scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

		if mask is not None:
			print('Using masking...')
			scaled_attention_logits = self.get_mask(scaled_attention_logits, mask)

		# softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
		attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
		scaled_attention = tf.matmul(attention_weights, V_c, name='Z')  # (..., seq_len_q, depth_v)

		scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

		concat_attention = tf.reshape(scaled_attention,
										(batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

		output = self.dense(concat_attention)
		
		return output, attention_weights

	def get_input_embedding(self, emb_x, pe_t, dropout, layers_outputs, training):
		
		if len(layers_outputs) == 0:
			if self.residual_type == 'residual_in_last_attblock':
				x = dropout(emb_x, training=training)
			else:
				if self.pe_func_name == 'pe_tm':
					alpha, beta = pe_t
					x = emb_x * alpha + beta
				else:
					x = emb_x + pe_t

				x = dropout(x, training=training)
				
		else:
			x = layers_outputs[-1]
		
		return x


	def get_mask(self, scaled_attention_logits, mask):
		steps = tf.shape(scaled_attention_logits)[2]
		mask_rshp = tf.tile(mask, [1,1,steps])
		mask_rshp += tf.transpose(mask_rshp, [0,2,1])
		mask_rshp = tf.minimum(1., mask_rshp)
		mask_rshp = tf.expand_dims(mask_rshp, 1)
		scaled_attention_logits += (mask_rshp*-1e9) 
		
		return scaled_attention_logits


	def get_p2p_term(self, pe_t, i_layer, num_layers, Wq_pos_Wk_pos, batch_size):
		if self.residual_type == 'residual_in_all_attblocks':
			Wq_pos = self.wq(pe_t)
			Wk_pos = self.wk(pe_t)
			Wq_pos = self.split_heads(Wq_pos, batch_size, name='Q_p') 
			Wk_pos = self.split_heads(Wk_pos, batch_size, name='K_p') 
			Wq_pos_Wk_pos = tf.matmul(Wq_pos, Wk_pos, transpose_b=True)

		elif self.residual_type == 'residual_in_last_attblock':
			if i_layer == num_layers-1:
				Wq_pos = self.wq(pe_t)
				Wk_pos = self.wk(pe_t)
				Wq_pos = self.split_heads(Wq_pos, batch_size, name='Q_p')
				Wk_pos = self.split_heads(Wk_pos, batch_size, name='K_p') 
				Wq_pos_Wk_pos = tf.matmul(Wq_pos, Wk_pos, transpose_b=True)

		else:
			raise TypeError("You wrote a residual type which is not created")

		return Wq_pos_Wk_pos


	def get_config(self):
		base_config = super().get_config()
		config = {
			"head_dim": self.head_dim,
			"num_heads": self.num_heads,
		}
		return {**base_config, **config}