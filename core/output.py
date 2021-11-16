import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense


class RegLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reg_layer = Dense(1, name='RegLayer')
		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		x = self.bn_0(inputs)
		x = self.reg_layer(x)
		return x
