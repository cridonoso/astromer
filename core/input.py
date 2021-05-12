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
	with tf.name_scope("InputFormat") as scope:
		with tf.name_scope("Separate_Vectors") as scope:
			batch_size      = tf.shape(data['serie_1'])[0]
			inp_dim 	    = tf.shape(data['serie_1'])[-1] -1 # we removed times
			times_1 = tf.slice(data['serie_1'], [0,0,0], [-1,-1, 1], 'times_1')
			magns_1 = tf.slice(data['serie_1'], [0,0,1], [-1,-1, 2], 'magns_1')
			times_2 = tf.slice(data['serie_2'], [0,0,0], [-1,-1, 1], 'times_2')
			magns_2 = tf.slice(data['serie_2'], [0,0,1], [-1,-1, 2], 'magns_2')

		with tf.name_scope("Serie_1") as scope:
			mask_1          = create_MASK_token(magns_1, frac=maskfrac)
			n_masked_1      = tf.reduce_sum(tf.cast(mask_1, tf.float32), 1,
											name='num_masked_1')
			serie_1, mask_1 = set_random(magns_1, mask_1, n_masked_1,
										 frac=randfrac)
			mask_1          = set_same(mask_1, n_masked_1, frac=samefrac)
			padd_mask       = create_padding_mask(magns_1, data['steps_1'])
			mask_1          = tf.math.logical_or(tf.squeeze(mask_1),
												 padd_mask,
												 name='LogicalOR')
			mask_1			= tf.cast(mask_1, tf.float32, 'BoolToFloat')

		with tf.name_scope("Serie_2") as scope:
			mask_2          = create_MASK_token(magns_2, frac=maskfrac)
			n_masked_2      = tf.reduce_sum(tf.cast(mask_2, tf.float32), 1,
							   name='num_masked_2')
			serie_2, mask_2 = set_random(magns_2, mask_2, n_masked_2,
									     frac=randfrac)
			mask_2          = set_same(mask_2, n_masked_2, frac=samefrac)
			padd_mask       = create_padding_mask(magns_2, data['steps_2'])
			mask_2          = tf.math.logical_or(tf.squeeze(mask_2),
												 padd_mask,
												 name='LogicalOR2')
			mask_2			= tf.cast(mask_2, tf.float32, 'BoolToFloat2')

		with tf.name_scope("NSP") as scope:
			indices      = tf.random.categorical(
								tf.math.log([[1-nppfrac, nppfrac]]),
								batch_size,
								name='NSRandomVector')
			indices      = tf.reshape(indices, [batch_size, 1, 1])

			if not use_random:
				indices = tf.zeros_like(indices)

			# concat values and mask to shufle them
			serie_mask = tf.concat([serie_1, tf.expand_dims(mask_1, 2)], 2,
									name='SerieMask_1')
			serie_mask_shuffled = tf.random.shuffle(serie_mask, name='Shuffle')

			rand_samples = tf.slice(serie_mask_shuffled, [0,0,0], [-1,-1, inp_dim],
									name='ShuffledSamples')
			rand_masks   = tf.slice(serie_mask_shuffled, [0,0,inp_dim], [-1,-1,1],
									name='ShuffledMasks')

			with tf.name_scope("BuildNewSerie") as scope:
				next_portion = tf.where(indices==1, rand_samples, serie_2,
									    name='RandomizedSerie')
				next_mask    = tf.where(indices==1, rand_masks,
										tf.expand_dims(mask_2, 2),
										name='RandomizedMask')
				next_mask    = tf.squeeze(next_mask)

		# Create input tokens
		sep_tokn = [[[sep_token]]]
		sep_tokn = tf.tile(sep_tokn, [batch_size, 1, inp_dim], name='SepTokens')
		cls_tokn = [[[cls_token]]]
		cls_tokn = tf.tile(cls_tokn, [batch_size, 1, inp_dim], name='ClsTokens')

		with tf.name_scope("CreateInput") as scope:
			serie_1      = standardize(serie_1, only_magn=True)
			next_portion = standardize(next_portion, only_magn=True)
			inputs       = tf.concat([cls_tokn,
									  serie_1,
									  sep_tokn,
									  next_portion,
									  sep_tokn], 1)

		with tf.name_scope("CreateTarget") as scope:
			if finetuning:
				indices = tf.reshape(data['label'], [-1, 1, 1])

			cls_label = tf.tile(indices, [1, 1, inp_dim]) # True NPP label
			cls_label = tf.cast(cls_label, tf.float32)

			target 	  = tf.concat([serie_1,
								   sep_tokn,
								   next_portion,
								   sep_tokn], 1)

		with tf.name_scope("CreateInputMask") as scope:
			tokn_mask  = tf.ones([batch_size, 1], dtype=tf.float32)
			inp_mask  = tf.concat([tokn_mask,
					               mask_1,
								   tokn_mask,
								   tf.cast(next_mask, tf.float32),
								   tokn_mask], 1)
			with tf.name_scope("ReshapeMask") as scope:
				len_mask  = tf.shape(inp_mask)[1]
				inp_mask  = tf.tile(inp_mask, [1, len_mask])
				inp_mask  = tf.reshape(inp_mask, [tf.shape(inp_mask)[0],
												  len_mask,
												  len_mask])
				inp_mask  = tf.expand_dims(inp_mask, 1)

		with tf.name_scope("CreateTargetMask") as scope:
			tar_mask  = tf.concat([tf.cast(mask_1, tf.float32),
								   tokn_mask-1,
								   tf.cast(next_mask, tf.float32),
								   tokn_mask-1], 1)

		with tf.name_scope("GetTimes") as scope:
			min_t1  = tf.expand_dims(tf.reduce_min(times_1, 1), 2, name='min_t1')
			times_1 = tf.math.subtract(times_1, min_t1, name='t1_zero_day')
			last    = tf.slice(times_1, [0,tf.shape(times_1)[1]-1,0], [-1, 1,-1],
							   name='last_t1')

			min_t2 = tf.expand_dims(tf.reduce_min(times_2, 1), 2, name='min_t2')
			times_2 = tf.math.subtract(times_2, min_t2, name='t2_zero_day')
			times_2 = tf.math.add(times_2, [[[1.]]], name='shift_t2_a_day')
			times_2 = tf.math.add(times_2, last, name='shift_t2_after_t1')

			times  = tf.concat([tokn_mask,
							    tf.squeeze(times_1),
							    tokn_mask,
							    tf.squeeze(times_2),
							    tokn_mask], 1)
			times = tf.expand_dims(times, 2)

		std = tf.slice(inputs, [0,1,1], [-1, -1, 1], name='GetSTD')
		weigths = tf.math.reciprocal_no_nan(std, 'loss_weights')
		weigths = normalize(weigths)

		inputs  = tf.slice(inputs, [0,0,0], [-1, -1, 1], name='GetMagns')


		inp_dict = {'values': inputs,
					'mask':inp_mask,
					'times': times}

		tar_dict = {'x_true': target,
			    	'y_true': cls_label,
					'x_mask':tar_mask,
					'weigths':weigths}

		return inp_dict, tar_dict
