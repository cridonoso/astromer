import tensorflow as tf 

from tensorflow.keras.layers import Input, Layer
from core.masking import create_mask, concat_mask


class InputLayer(Layer):
	def __init__(self, frac=0.15, rand_frac=0.1, same_frac=0.1, sep_token=102., **kwargs):
		super(InputLayer, self).__init__(**kwargs)
		self.frac = frac # TOKENS
		self.randfrac  = rand_frac # Replace by random magnitude
		self.samefrac  = same_frac # Replace by the same magnitude
		self.sep_token = sep_token

	def call(self, data, training=False):
		
		mask_1_tar, mask_1_inp = create_mask(data[0])

		mask_2_tar, mask_2_inp = create_mask(x2, length=length, frac=self.frac,
		                    frac_random=self.randfrac, frac_same=self.samefrac)


		mask_inp = concat_mask(mask_1_inp, mask_2_inp, cls_true, sep=self.sep_token)
		mask_tar = concat_mask(mask_1_tar, mask_2_tar, cls_true, sep=self.sep_token, reshape=False)

		batch_size = tf.shape(x1)[0]
		inp_dim = tf.shape(x1)[-1]

		cls_true = tf.expand_dims(cls_true, 2) # (1, 1, 1)
		cls_true = tf.tile(cls_true, [1, 1, inp_dim], name='CLSTokens')

		sep_tokn = [[[self.sep_token]]] # (1,1,1)
		sep = tf.tile(sep_tokn, [batch_size, 1, inp_dim], name='SepTokens')

		inputs = tf.concat([cls_true, x1, sep, x2], 1, name='NetworkInput')


		return inputs, mask_inp, mask_tar