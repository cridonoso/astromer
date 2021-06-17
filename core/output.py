import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense


class RegLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reg_layer_0 = Dense(64, name='RegLayer_0')
		self.reg_layer_1 = Dense(1, name='RegLayer_1')

		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		x = self.reg_layer_0(inputs)
		x = self.bn_0(x)
		x = self.reg_layer_1(x)
		return x

class ClfLayer(Layer):
	def __init__(self, num_cls=2, dropout=0.4, **kwargs):
		super().__init__(**kwargs)
		self.drop_0 = tf.keras.layers.Dropout(dropout)
		self.drop_1 = tf.keras.layers.Dropout(dropout)

		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.bn_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

		self.cls_layer_0 = Dense(64, name='CLFLayer_0')
		self.cls_layer_1 = Dense(num_cls, name='CLFLayer_1')

	def call(self, inputs):
		cls_prob = self.cls_layer_0(inputs)
		cls_prob = self.drop_0(cls_prob)
		cls_prob = self.bn_0(cls_prob)
		cls_prob = self.cls_layer_1(cls_prob)
		cls_prob = self.drop_1(cls_prob)
		cls_prob = self.bn_1(cls_prob)

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
