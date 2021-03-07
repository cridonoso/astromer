import tensorflow as tf 

from tensorflow.keras.layers import Input, Layer
from tensorflow.keras import Model

from core.encoder import Encoder
from core.decoder import Decoder
from core.masking import create_mask, concat_mask


class InputComposer(Layer):
    def __init__(self, frac=0.15, rand_frac=0.1, same_frac=0.1, sep_token=102., **kwargs):
        super(InputComposer, self).__init__(**kwargs)
        self.frac = frac # TOKENS 
        self.randfrac = rand_frac # Replace by random magnitude
        self.samefrac = same_frac # Replace by the same magnitude
        self.sep_token = sep_token

    def call(self, data):
        x1, x2, length, cls_true = data

        mask_1_tar, mask_1_inp = create_mask(x1)
        
        mask_2_tar, mask_2_inp = create_mask(x2, length=length, frac=self.frac, 
                            frac_random=self.randfrac, frac_same=self.samefrac)

        mask_inp = concat_mask(mask_1_inp, mask_2_inp, cls_true, sep=self.sep_token)
        mask_tar = concat_mask(mask_1_tar, mask_2_tar, cls_true, sep=self.sep_token, reshape=False)

        cls_true = tf.tile(tf.expand_dims(cls_true, 2), [1, 1, tf.shape(x1)[-1]], name='CLSTokens')
        sep = tf.tile([[[self.sep_token]]], [tf.shape(x1)[0],1,tf.shape(x1)[-1]], name='SepTokens')
        inputs = tf.concat([cls_true, x1, sep, x2], 1, name='NetworkInput')

        return inputs, mask_inp, mask_tar

class CustomDense(Layer):
    def __init__(self, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.reg_layer = tf.keras.layers.Dense(1)
        self.cls_layer = tf.keras.layers.Dense(2)

    def call(self, inputs):
        logist_rec = tf.slice(inputs, [0,1,0], [-1, -1, -1])
        logist_cls = tf.slice(inputs, [0,0,0], [-1, 1, -1])
        cls_prob = self.cls_layer(logist_cls)
        cls_prob = tf.transpose(cls_prob, [0,2,1])
        reconstruction = self.reg_layer(logist_rec)
        final_output = tf.concat([cls_prob, reconstruction], 
                                 axis=1)

        return final_output

class ASTROMER(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, rate=0.1):
        super(ASTROMER, self).__init__(name='ASTROMER')
        self.input_layer = InputComposer(name='BuildInput')
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate, name='Encoder')
        self.dense   = CustomDense(name='Dense')

    def model(self, batch_size):
        serie_1  = Input(shape=(202, 2), batch_size=batch_size, name='Serie1')
        serie_2  = Input(shape=(202, 2), batch_size=batch_size, name='Serie2')
        length_i = Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='TrueLength')
        class_i  = Input(shape=(1,), batch_size=batch_size, name='IsRandom')
        data = (serie_1, serie_2, length_i, class_i)
        return Model(inputs=data, outputs=self.call(data))

    def call(self, inputs, training=False):
        # inp, mask = inputs
        inp, mask_inp, mask_tar = self.input_layer(inputs)
        enc_output = self.encoder(inp, training, mask=mask_inp) 
        final_output = self.dense(enc_output)
        m = tf.concat([tf.expand_dims(mask_tar[:, 0], 1), mask_tar], 1)
        output_mask = tf.concat([final_output, tf.expand_dims(m, 2)], 2)
        
        return output_mask, inp 

    def train_step(self, data):
        cls_true = data[-1]
        with tf.GradientTape() as tape:
            output, inputs = self(data, training=True)
            t_loss = self.compiled_loss(inputs, output)
            
        gradients = tape.gradient(t_loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(inputs, output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        output, inputs = self(data, training=False)
        t_loss = self.compiled_loss(inputs, output)
        self.compiled_metrics.update_state(inputs, output)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, data):
        output, inputs = self(data, training=False)
        self.compiled_metrics.update_state(inputs, output)
        return {m.name: m.result() for m in self.metrics}