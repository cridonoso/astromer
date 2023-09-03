import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense


class TransformLayer(Layer):
	def __init__(self, **kwargs):
		super(TransformLayer, self).__init__(**kwargs)
		self.clf_layer = Dense(2, name='Classification')
		self.reg_layer = Dense(1, name='Reconstruction')
		
	def call(self, inputs):

		cls_token = tf.slice(inputs, [0, 0, 0], [-1, 1, -1], name='cls_token')
		rec_token = tf.slice(inputs, [0, 1, 0], [-1, -1, -1], name='rec_token')

		x_prob = self.clf_layer(cls_token)
		x_rec = self.reg_layer(rec_token)

		return {'nsp_label': x_prob, 'reconstruction':x_rec}

class RegLayer(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.reg_layer = Dense(1, name='RegLayer')
		self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

	def call(self, inputs):
		x = self.bn_0(inputs)
		x = self.reg_layer(x)
		return x