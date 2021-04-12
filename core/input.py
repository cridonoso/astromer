import tensorflow as tf 

from tensorflow.keras.layers import Input, Layer
from core.masking import create_MASK_token, create_padding_mask, set_random, set_same
from core.data import normalize, standardize

class InputLayer(Layer):
	def __init__(self, mask_frac=0.15, npp_frac=0.5, rand_frac=0.1, same_frac=0.1, sep_token=102., cls_token=101., **kwargs):
		super().__init__(**kwargs)
		self.maskfrac = mask_frac # TOKENS
		self.nppfrac   = npp_frac # next portion prediction
		self.randfrac  = rand_frac # Replace by random magnitude
		self.samefrac  = same_frac # Replace by the same magnitude
		self.sep_token = tf.cast(sep_token, tf.float32)
		self.cls_token = tf.cast(cls_token, tf.float32)

	def call(self, data, training=False):
		"""
		This function creates a BERT-like input for ASTROMER.
		i.e., 
			1.- add [SEP], [MASK], and [CLS] tokens
			2.- concat serie_1 and serie_2
			3.- serie_2 could be replaced by another random serie_2 within the batch
			4.- creates differnet masks	
		Args:
		    data (dictonary): A dictonary containing the current batch. 
		    				  Keys: 'serie_1', 'serie_2', 'steps_1', 'steps_1'
		    training (bool, optional): training varible
		
		Returns:
		    dictonary: a dictionary containing the new input format
		"""
		# First serie 
		mask_1     = create_MASK_token(data['serie_1'], frac=self.maskfrac)
		n_masked   = tf.reduce_sum(tf.cast(mask_1, tf.float32), 1)
		s1, mask_1 = set_random(data['serie_1'], mask_1, n_masked, frac=self.randfrac)
		mask_1     = set_same(mask_1, n_masked, frac=self.samefrac)
		padd_mask  = create_padding_mask(data['serie_1'], data['steps_1'])

		mask_1     = tf.math.logical_or(tf.squeeze(mask_1), padd_mask)	
		serie_1    = normalize(s1, only_time=True)

		# Second serie
		mask_2     = create_MASK_token(data['serie_2'], frac=self.maskfrac)
		n_masked   = tf.reduce_sum(tf.cast(mask_2, tf.float32), 1)
		s2, mask_2 = set_random(data['serie_2'], mask_2, n_masked, frac=self.randfrac)
		mask_2     = set_same(mask_2, n_masked, frac=self.samefrac)
		padd_mask  = create_padding_mask(data['serie_2'], data['steps_2'])
		mask_2     = tf.math.logical_or(tf.squeeze(mask_2), padd_mask)
		serie_2    = normalize(s2, only_time=True)


		# Next portion prediction 
		batch_size   = tf.shape(mask_2)[0]
		indices      = tf.random.categorical(tf.math.log([[1-self.nppfrac, self.nppfrac]]), batch_size)[0]
		indices      = tf.reshape(indices, [-1, 1, 1])
		rand_samples = tf.random.shuffle(serie_1)

		# Shifting time to force continous time
		inp_dim 	 = tf.shape(serie_1)[-1]
		ones_v  	 = tf.slice(serie_1, [0,tf.shape(serie_1)[1]-1,0],[-1, 1, 1])
		zeros_v 	 = tf.zeros([batch_size, 1, inp_dim-1])
		shift 		 = tf.concat([ones_v, zeros_v], 2)	
		next_portion = tf.where(indices==1, rand_samples+shift, serie_2+shift) 	

		# Create input format		
		sep_tokn = [[[self.sep_token]]] 
		sep_tokn = tf.tile(sep_tokn, [batch_size, 1, inp_dim], name='SepTokens')

		cls_tokn = [[[self.cls_token]]] 
		cls_tokn = tf.tile(cls_tokn, [batch_size, 1, inp_dim], name='ClsTokens')

		inputs   = tf.concat([cls_tokn, serie_1, sep_tokn, next_portion, sep_tokn], 1)

		# Adding true npp labels 
		cls_label = tf.tile(indices, [1, 1, inp_dim])
		cls_label = tf.cast(cls_label, tf.float32)
		target 	  = tf.concat([cls_label, serie_1, sep_tokn, next_portion, sep_tokn], 1)

		# Create input and target masks
		sep_tokn  = tf.slice(sep_tokn, [0, 0, 0], [-1, -1, 1])
		sep_tokn  = tf.ones_like(tf.reshape(sep_tokn, [-1, 1]), dtype=tf.float32)
		cls_label = tf.slice(cls_label, [0, 0, 0], [-1, -1, 1])
		cls_label = tf.ones_like(tf.reshape(cls_label, [-1, 1]), dtype=tf.float32)

		inp_mask  = tf.concat([cls_label, 
				               tf.cast(mask_1, tf.float32), 
							   sep_tokn, 
							   tf.cast(mask_2, tf.float32),
							   sep_tokn], 1)

		dim_mask  = tf.shape(inp_mask)[1]
		inp_mask  = tf.tile(inp_mask, [1, dim_mask])
		inp_mask  = tf.reshape(inp_mask, [tf.shape(inp_mask)[0], dim_mask, dim_mask])
		inp_mask  = tf.expand_dims(inp_mask, 1)

		tar_mask  = tf.concat([cls_label-1, 
							   tf.cast(mask_1, tf.float32), 
							   sep_tokn-1, 
							   tf.cast(mask_2, tf.float32),
							   sep_tokn-1], 1)
		
		# removing times from the input matrix
		inputs = tf.slice(inputs, [0,0,1], [-1, -1, 1])

		return {'inputs': inputs,
				'target': target,
				'inp_mask':inp_mask,
				'tar_mask':tar_mask}