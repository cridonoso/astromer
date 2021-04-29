import tensorflow as tf

from tensorflow.keras.layers import Input, Layer, Dense


class OutputLayer(Layer):
	def __init__(self, **kwargs):
		super(OutputLayer, self).__init__(**kwargs)
		self.reg_layer = Dense(1, name='RegressionLayer')
		self.cls_layer = Dense(2, name='ClassificationLayer')

	def call(self, inputs):
		logist_rec = tf.slice(inputs, [0,1,0], [-1, -1, -1],
							  name='RecontructionSplit')
		logist_cls = tf.slice(inputs, [0,0,0], [-1, 1, -1],
							  name='ClassPredictedSplit')
		cls_prob = self.cls_layer(logist_cls)
		cls_prob = tf.transpose(cls_prob, [0,2,1],
								name='CategoricalClsPred')
		reconstruction = self.reg_layer(logist_rec)
		final_output = tf.concat([cls_prob, reconstruction],
		                         axis=1,
		                         name='ConcatClassRec')
		return final_output
