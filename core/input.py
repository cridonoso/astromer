import tensorflow as tf

from core.data import normalize, standardize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from core.masking import get_padding_mask


@tf.function
def input_format(batch, clstkn=-99, septkn=-98):
	"""
	This function adds [SEP] and [CLS] tokens to the input batch
	"""
	inp_shp = tf.shape(batch['input'])

	cls_tkn = tf.cast([[[clstkn]]], dtype=tf.float32)
	cls_tkn = tf.tile(cls_tkn, [inp_shp[0], 1, inp_shp[-1]])

	sep_tkn = tf.cast([[[septkn]]], dtype=tf.float32)
	sep_tkn = tf.tile(sep_tkn, [inp_shp[0], 1, inp_shp[-1]])

	x1, x2 = tf.split(batch['input'], 2, 1)
	t1, t2 = tf.split(batch['times'], 2, 1)

	padd = get_padding_mask(inp_shp[1], batch['length'])
	padd = tf.reshape(padd, [-1, inp_shp[1], 1])

	mask_tkn = tf.cast([[[0]]], dtype=tf.float32)
	mask_tkn = tf.tile(mask_tkn, [inp_shp[0], 1, 1])

	mask = batch['mask_in'] + padd
	m1, m2 = tf.split(mask, 2, 1)

	inputs = tf.concat([cls_tkn, x1, sep_tkn, x2, sep_tkn], 1)
	times  = tf.concat([cls_tkn, t1, sep_tkn, t2, sep_tkn], 1)
	mask   = tf.concat([mask_tkn, m1, mask_tkn, m2, mask_tkn], 1)

	return inputs, times, mask
