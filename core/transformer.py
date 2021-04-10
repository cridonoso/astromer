import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from core.input   import InputLayer
from core.output  import OutputLayer
from core.encoder import Encoder
from core.decoder import Decoder



class ASTROMER(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, pe_input, rate=0.1, inp_dim=4, ):
        super(ASTROMER, self).__init__(name='ASTROMER')
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.d_model    = d_model 
        self.num_heads  = num_heads
        self.dff        = dff
        self.pe_input   = pe_input
        self.rate       = rate
        self.inp_dim    = inp_dim

        self.input_layer = InputLayer(name='BuildInput')

        self.encoder     = Encoder(num_layers, 
                                   d_model, 
                                   num_heads, 
                                   dff,
                                   pe_input, 
                                   rate, 
                                   inp_dim=inp_dim, 
                                   name='Encoder')

        self.dense       = OutputLayer(name='Dense')

    def call(self, inputs, training=False):
        ids, serie_1, serie_2, label = inputs

        # Join series by [SEP]
        inp_vector, mask_input, mask_target = self.input_layer(serie_1, 
                                                               serie_2)


        # enc_output = self.encoder(inp, training, mask=mask_inp)
        # final_output = self.dense(enc_output)
        # m = tf.concat([tf.expand_dims(mask_tar[:, 0], 1), mask_tar], 1, name='RepeatClassMask')
        # output_mask = tf.concat([final_output, tf.expand_dims(m, 2)], 2, name='ConcatPredsAndMask')
        # return output_mask, inp
        return inp_vector

    def model(self, batch_size):
        serie_1  = Input(shape=(100, 3), batch_size=batch_size, name='Serie1')
        serie_2  = Input(shape=(100, 3), batch_size=batch_size, name='Serie2')

        length_i = Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='TrueLength')
        id_i     = Input(shape=(), batch_size=batch_size, dtype=tf.string, name='ID')
        lab_cls  = Input(shape=(), batch_size=batch_size, name='label')

        data = (id_i, serie_1, serie_2,lab_cls)
        return Model(inputs=data, outputs=self.call(data))
    
    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    def train_step(self, data):
        with tf.GradientTape() as tape:
            output, inputs = self(data, training=True)
            t_loss = self.compiled_loss(inputs, output, sample_weight=data[-1])

        gradients = tape.gradient(t_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(inputs, output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        output, inputs = self(data, training=False)
        t_loss = self.compiled_loss(inputs, output, sample_weight=data[-1])
        self.compiled_metrics.update_state(inputs, output)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        y_pred, y_true = self(data, training=False)

        rec_pred = tf.slice(y_pred, [0,2,0], [-1, -1, 1])
        rec_mask = tf.slice(y_pred, [0,2,1], [-1, -1, 1])
        rec_true = tf.slice(y_true, [0,1,1], [-1, -1, -1])
        rec_times = tf.slice(y_true, [0,1,0], [-1, -1, 1])
        cls_pred = tf.argmax(tf.slice(y_pred, [0,0,0], [-1, 2, 1]), 1)
        cls_true = tf.slice(y_true, [0,0,0], [-1, 1, 1])

        return tf.squeeze(rec_pred), tf.squeeze(rec_mask), \
               tf.squeeze(rec_true), tf.squeeze(rec_times), \
               tf.squeeze(cls_pred), tf.squeeze(cls_true)

    def get_attention(self, data):
        all_vectors = []
        for d in data:
            inp, mask_inp, mask_tar = self.input_layer(d)
            enc_output = self.encoder(inp, False, mask=mask_inp)
            all_vectors.append(enc_output)

        return all_vectors
