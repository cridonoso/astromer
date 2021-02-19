import tensorflow as tf 

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from core.encoder import Encoder
from core.decoder import Decoder
from core.masking import create_masks


class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff, 
                                input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                                target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def compile(self, optimizer, loss_function, **kwargs):
        super(Transformer, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss_function = loss_function

    def model(self, batch_size):
        inp = Input(shape=(100), batch_size=batch_size, name='Input')
        tar = Input(shape=(100
        ), batch_size=batch_size, name='Target')
        tar_inp  = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
        inputs = (inp, tar_inp, enc_padding_mask, combined_mask, dec_padding_mask)
        return Model(inputs=inputs, outputs=self.call(inputs, training=True))


    def call(self, inputs, training=False):
        inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask = inputs
        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def train_step(self, data):
        inp, tar = data
        tar_inp  = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = self((inp, tar_inp,  
                                  enc_padding_mask, 
                                  combined_mask, 
                                  dec_padding_mask), 
                                  training=True)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(tar_real, predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inp, tar = data
        tar_inp  = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        predictions, _ = self((inp, tar_inp,  
                               enc_padding_mask, 
                               combined_mask, 
                               dec_padding_mask), 
                               training=False)
        loss = self.loss_function(tar_real, predictions)
        self.compiled_metrics.update_state(tar_real, predictions)

        return {m.name: m.result() for m in self.metrics}

class MiniTransformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                target_vocab_size, pe_input, rate=0.1):
        super(MiniTransformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                                input_vocab_size, pe_input, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def compile(self, optimizer, loss_function, **kwargs):
        super(MiniTransformer, self).compile(**kwargs)
        self.optimizer = optimizer
        self.loss_function = loss_function

    def model(self, batch_size):
        inp = Input(shape=(66), batch_size=batch_size, name='Input')
        mask = Input(shape=(1, 66, 66), batch_size=batch_size, name='Mask')

        return Model(inputs=(inp, mask), outputs=self.call((inp, mask)))


    def call(self, inputs, training=False):
        inp, mask = inputs
        enc_output = self.encoder(inp, training, mask) 
        final_output = self.final_layer(enc_output)
        return final_output

    def train_step(self, data):
        inp, tar = data
        inp_mask = create_masks(inp)

        with tf.GradientTape() as tape:
            predictions = self((inp, inp_mask), training=True)
            loss = self.loss_function(tar, predictions)

        gradients = tape.gradient(loss, self.trainable_variables)    
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(tar, predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inp, tar = data
        inp_mask = create_masks(inp)

        predictions = self((inp, inp_mask), training=True)

        loss = self.loss_function(tar, predictions)
        self.compiled_metrics.update_state(tar, predictions)

        return {m.name: m.result() for m in self.metrics}

    
