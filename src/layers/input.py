import tensorflow as tf

from tensorflow.keras.layers import Layer

class AddMSKToken(Layer):
    """ Create MSK token """
    def __init__(self, 
                 trainable=True,
                 on=['input'],
                 **kwargs):

        super().__init__(**kwargs)

        self.trainable = trainable
        self.on = on

    def build(self, input_shape):
        self.msk_token = tf.Variable(
                        initial_value=tf.constant([[0.]]),
                        dtype=tf.float32,
                        trainable=self.trainable,)

        self.msk_token = tf.tile(self.msk_token, [input_shape['input'][1],1])

    def call(self, inputs, training):
        for key in self.on:
            partial = tf.multiply(inputs[key], 1.-inputs['mask_in'])
            partial_mask = tf.multiply(inputs['mask_in'], self.msk_token)
            partial = tf.add(partial, partial_mask)
            inputs[key] = partial 
        return inputs


