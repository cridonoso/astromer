import tensorflow as tf

from tensorflow.keras.layers import Input, Layer
from core.masking import create_MASK_token, create_padding_mask, set_random, set_same
from core.data import normalize, standardize, get_delta


def input_format(data,
				 maskfrac=0.15,
				 randfrac=0.1,
				 samefrac=0.1,
				 nppfrac=0.5,
				 sep_token=102.,
				 cls_token=101.,
				 use_random=True,
				 finetuning=False):
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
	mask_1     = create_MASK_token(data['serie_1'], frac=maskfrac)
	n_masked   = tf.reduce_sum(tf.cast(mask_1, tf.float32), 1)
	s1, mask_1 = set_random(data['serie_1'], mask_1, n_masked, frac=randfrac)
	mask_1     = set_same(mask_1, n_masked, frac=samefrac)
	padd_mask  = create_padding_mask(data['serie_1'], data['steps_1'])

	mask_1     = tf.math.logical_or(tf.squeeze(mask_1), padd_mask)
	serie_1    = s1 # normalize(s1, only_time=True)

	# Second serie
	mask_2     = create_MASK_token(data['serie_2'], frac=maskfrac)
	n_masked   = tf.reduce_sum(tf.cast(mask_2, tf.float32), 1)
	s2, mask_2 = set_random(data['serie_2'], mask_2, n_masked, frac=randfrac)
	mask_2     = set_same(mask_2, n_masked, frac=samefrac)
	padd_mask  = create_padding_mask(data['serie_2'], data['steps_2'])
	mask_2     = tf.math.logical_or(tf.squeeze(mask_2), padd_mask)
	serie_2    = s2 #normalize(s2, only_time=True)

	# Next portion prediction
	batch_size   = tf.shape(mask_2)[0]
	indices      = tf.random.categorical(tf.math.log([[1-nppfrac, nppfrac]]), batch_size)[0]
	indices      = tf.reshape(indices, [-1, 1, 1])

	if not use_random:
		indices = tf.zeros_like(indices)

	# concat values and mask to shufle them
	serie_mask = tf.concat([serie_1, tf.expand_dims(tf.cast(mask_1, tf.float32), 2)], 2)
	serie_mask_shuffled = tf.random.shuffle(serie_mask)
	inp_dim 	 = tf.shape(serie_1)[-1] # number of parameters in the serie
	rand_samples = tf.slice(serie_mask_shuffled, [0,0,0], [-1,-1,inp_dim])
	rand_masks   = tf.slice(serie_mask_shuffled, [0,0,inp_dim], [-1,-1,1])
	rand_masks   = tf.cast(rand_masks, tf.bool)

	next_portion = tf.where(indices==1, rand_samples, serie_2)
	next_mask    = tf.where(indices==1, rand_masks, tf.expand_dims(mask_2, 2))
	next_mask    = tf.squeeze(next_mask)

	# Create input tokens
	sep_tokn = [[[sep_token]]]
	sep_tokn = tf.tile(sep_tokn, [batch_size, 1, inp_dim], name='SepTokens')

	cls_tokn = [[[cls_token]]]
	cls_tokn = tf.tile(cls_tokn, [batch_size, 1, inp_dim], name='ClsTokens')

	# =============================== INPUT ================================
	serie_1      = standardize(serie_1, only_magn=True)
	next_portion = standardize(next_portion, only_magn=True)
	inputs       = tf.concat([cls_tokn,
							  serie_1,
							  sep_tokn,
							  next_portion,
							  sep_tokn], 1)

	# =============================== TARGET ===============================
	if finetuning:
		indices = tf.reshape(data['label'], [-1, 1, 1])

	cls_label = tf.tile(indices, [1, 1, inp_dim]) # True NPP label
	cls_label = tf.cast(cls_label, tf.float32)

	target 	  = tf.concat([serie_1,
						   sep_tokn,
						   next_portion,
						   sep_tokn], 1)
	# ======================================================================

	sep_tokn_mask  = tf.slice(sep_tokn, [0, 0, 0], [-1, -1, 1])
	sep_tokn_mask  = tf.zeros_like(tf.reshape(sep_tokn_mask, [-1, 1]), dtype=tf.float32)
	cls_tokn_mask  = tf.slice(cls_label, [0, 0, 0], [-1, -1, 1])
	cls_tokn_mask  = tf.zeros_like(tf.reshape(cls_tokn_mask, [-1, 1]), dtype=tf.float32)

	# ============================= INPUT MASK =============================
	inp_mask  = tf.concat([cls_tokn_mask,
			               tf.cast(mask_1, tf.float32),
						   sep_tokn_mask,
						   tf.cast(next_mask, tf.float32),
						   sep_tokn_mask], 1)
	dim_mask  = tf.shape(inp_mask)[1]
	inp_mask  = tf.tile(inp_mask, [1, dim_mask])
	inp_mask  = tf.reshape(inp_mask, [tf.shape(inp_mask)[0], dim_mask, dim_mask])
	inp_mask  = tf.expand_dims(inp_mask, 1)

	# ============================ TARGET MASK =============================
	tar_mask  = tf.concat([tf.cast(mask_1, tf.float32),
						   sep_tokn_mask,
						   tf.cast(next_mask, tf.float32),
						   sep_tokn_mask], 1)

	# ============================= TIMES ==================================
	time_1 = tf.slice(inputs, [0,1,0], [-1, tf.shape(serie_1)[1], 1])
	time_2 = tf.slice(inputs, [0,tf.shape(serie_1)[1]+2,0], [-1, tf.shape(serie_1)[1], 1])

	time_1 = time_1 - tf.expand_dims(tf.reduce_min(time_1, 1), 2)
	dt = tf.expand_dims(tf.reduce_mean(time_1, 1), 2)
	last = tf.slice(time_1, [0,tf.shape(time_1)[1]-1,0], [-1, 1,-1])

	time_2 = time_2 - tf.expand_dims(tf.reduce_min(time_2, 1), 2)
	time_2 = time_2 + last + [[[1.]]]

	times  = tf.concat([cls_tokn_mask,
					    tf.squeeze(time_1),
					    sep_tokn_mask,
					    tf.squeeze(time_2),
					    sep_tokn_mask], 1)
	times = tf.expand_dims(times, 2)

	# Drop times and uncertainties from the input vector
	std     = tf.slice(inputs, [0,1,2], [-1, -1, 1])
	weigths = tf.math.reciprocal_no_nan(std)

	inputs  = tf.slice(inputs, [0,0,1], [-1, -1, 1])

	inp_dict = {'values': inputs,
				'mask':inp_mask,
				'times': times}

	tar_dict = {'x_true': target,
		    	'y_true': cls_label,
				'x_mask':tar_mask,
				'weigths':weigths}

	return inp_dict, tar_dict
