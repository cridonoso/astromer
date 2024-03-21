import tensorflow as tf

from tensorflow.keras.layers import Layer

class AddMSKToken(Layer):
    """ Create MSK token """
    def __init__(self, 
                 trainable=True,
                 window_size=200,
                 on=['input'],
                 **kwargs):

        super().__init__(**kwargs)

        self.trainable = trainable
        self.on = on
        self.window_size = window_size

    def build(self, input_shape):
        self.msk_token = tf.Variable(
                        initial_value=tf.constant([[0.]]),
                        dtype=tf.float32,
                        trainable=self.trainable,)

    def call(self, inputs, training):
        msk_token = tf.tile(self.msk_token, [self.window_size, 1])
        print(msk_token)
        for key in self.on:
            partial = tf.multiply(inputs[key], 1.-inputs['mask_in'])
            partial_mask = tf.multiply(inputs['mask_in'], msk_token)
            partial = tf.add(partial, partial_mask)
            inputs[key] = partial 
        return inputs


