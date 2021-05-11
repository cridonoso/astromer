import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense


class RegLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reg_layer = Dense(1, name='RegressionLayer')

	def call(self, inputs):
		reconstruction = self.reg_layer(inputs)

		return reconstruction

class ClfLayer(Layer):
	def __init__(self, num_cls=2, **kwargs):
		super().__init__(**kwargs)
		self.cls_layer = Dense(num_cls, name='ClassificationLayer')

	def call(self, inputs):
		cls_prob = self.cls_layer(inputs)

		return cls_prob


class SplitLayer(Layer):
	def __init__(self, num_cls=2, **kwargs):
		super().__init__(**kwargs)

	def call(self, inputs):
		logist_cls = tf.slice(inputs, [0,0,0], [-1, 1, -1],
							  name='z_npp')
		logist_rec = tf.slice(inputs, [0,1,0], [-1, -1, -1],
							  name='z_rec')

		return logist_cls, logist_rec
