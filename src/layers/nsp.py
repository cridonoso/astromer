import tensorflow as tf

from tensorflow.keras.layers import Layer

class ClassToken(Layer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def build(self, input_shape):
		w_init = tf.random_normal_initializer()
		self.w = tf.Variable(
			initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
			trainable = True
		)

	@tf.function
	def call(self, inputs):
		batch_size = tf.shape(inputs)[0]
		hidden_dim = self.w.shape[-1]

		cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
		cls = tf.cast(cls, dtype=inputs.dtype)
		return cls