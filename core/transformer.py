import tensorflow as tf 

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from core.encoder import Encoder
from core.decoder import Decoder
from core.masking import create_mask, concat_mask


def create_input(x1, x2, cls):
    cls = tf.tile(tf.expand_dims(cls, 2), [1, 1, tf.shape(x1)[-1]])
    sep = tf.tile([[[102.]]], [tf.shape(x1)[0],1,tf.shape(x1)[-1]])
    inputs = tf.concat([cls, x1, sep, x2], 1)
    return inputs

class ASTROMER(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, rate=0.1):
        super(ASTROMER, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, pe_input, rate)

        self.final_layer = tf.keras.layers.Dense(1)

    def compile(self, optimizer, rec_loss, cls_loss, **kwargs):
        super(ASTROMER, self).compile(**kwargs)
        self.optimizer = optimizer
        self.rec_loss = rec_loss
        self.cls_loss = cls_loss

    def model(self, batch_size):
        inp = Input(shape=(202, 2), batch_size=batch_size, name='Input')
        mask = Input(shape=(1, 202, 202), batch_size=batch_size, name='Mask')
        return Model(inputs=(inp, mask), outputs=self.call((inp, mask)))


    def call(self, inputs, training=False):
        inp, mask = inputs
        enc_output = self.encoder(inp, training, mask=mask) 
        final_output = self.final_layer(enc_output)
        cls_pred = tf.slice(final_output, [0,0,0], [-1, 1, -1])
        rec_pred = tf.slice(final_output, [0,1,0], [-1, -1, -1])

        return rec_pred, cls_pred

    def train_step(self, data):
        x1, x2, length, cls_true = data

        mask1 = create_mask(x1)
        mask2 = create_mask(x2, length)
        mask = concat_mask(mask1, mask2, cls_true)

        inputs = create_input(x1, x2, cls_true)

        with tf.GradientTape() as tape:
            rec_pred, cls_pred = self((inputs, mask), training=True)
            rec_true = tf.slice(inputs, [0,1,1], [-1, -1, 1])
            loss_cls = self.cls_loss(tf.squeeze(cls_true), tf.squeeze(cls_pred))
            loss_rec = self.rec_loss(rec_true, rec_pred)
            loss_rec = tf.reduce_sum(loss_rec, 1)
            loss = loss_cls+loss_rec

        gradients = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(rec_true, rec_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x1, x2, length, cls_true = data

        mask1 = create_mask(x1)
        mask2 = create_mask(x2, length)
        mask = concat_mask(mask1, mask2, cls_true)

        inputs = create_input(x1, x2, cls_true)

        rec_pred, cls_pred = self((inputs, mask), training=True)
        rec_true = tf.slice(inputs, [0,1,1], [-1, -1, 1])
        loss_cls = self.cls_loss(tf.squeeze(cls_true), tf.squeeze(cls_pred))
        loss_rec = self.rec_loss(rec_true, rec_pred)
        loss_rec = tf.reduce_sum(loss_rec, 1)
        loss = loss_cls+loss_rec

        self.compiled_metrics.update_state(rec_true, rec_pred)

        return {m.name: m.result() for m in self.metrics}

    
    def predict_step(self, data):
        inp, tar = data
        inp_mask = create_masks(inp)

        predictions = self((inp, inp_mask), training=True)
        index = tf.argmax(predictions, 2)
        return tokenizers.en.detokenize(index), tokenizers.en.detokenize(tar)