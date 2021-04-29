import tensorflow as tf

from tensorflow.keras.layers import Input
from tensorflow.keras import Model

from core.input   import InputLayer
from core.output  import OutputLayer
from core.encoder import Encoder
from core.decoder import Decoder



class ASTROMER(Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 rate=0.1,
                 base=10000,
                 mask_frac=0.15,
                 npp_frac=0.5,
                 rand_frac=0.1,
                 same_frac=0.1,
                 sep_token=102.,
                 cls_token=101.):

        super().__init__(name='ASTROMER')
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.dff        = dff
        self.rate       = rate
        self.base       = base
        self.mask_frac  = mask_frac
        self.npp_frac   = npp_frac
        self.rand_frac  = rand_frac
        self.same_frac  = same_frac
        self.sep_token  = sep_token
        self.cls_token  = cls_token

        self.input_layer = InputLayer(mask_frac=mask_frac,
                                      npp_frac=npp_frac,
                                      rand_frac=rand_frac,
                                      same_frac=same_frac,
                                      sep_token=sep_token,
                                      cls_token=cls_token,
                                      name='BuildInput')

        self.encoder     = Encoder(num_layers,
                                   d_model,
                                   num_heads,
                                   dff,
                                   base,
                                   rate,
                                   name='Encoder')

        self.output_layer = OutputLayer(name='Dense')

    def call(self, inputs, training=False):
        # Join series by [SEP]
        in_dict = self.input_layer(inputs)
        enc_output = self.encoder(in_dict, training)
        final_output = self.output_layer(enc_output)

        # we need to adjust our mask to match the current dimensionality
        in_dict['tar_mask'] = tf.concat([tf.expand_dims(in_dict['tar_mask'][:, 0], 1),
                                         in_dict['tar_mask']], 1, name='RepeatClassMask')

        output_mask = tf.concat([final_output, tf.expand_dims(in_dict['tar_mask'], 2)],
                                2,
                                name='ConcatPredsAndMask')

        return output_mask, in_dict['target']

    def model(self, batch_size):
        serie_1  = Input(shape=(50, 3), batch_size=batch_size, name='Serie1')
        serie_2  = Input(shape=(50, 3), batch_size=batch_size, name='Serie2')

        steps_1 = Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='steps_1')
        steps_2 = Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='steps_2')

        data = {'serie_1':serie_1, 'serie_2':serie_2,
                'steps_1':steps_1, 'steps_2':steps_2}

        return Model(inputs=data, outputs=self.call(data))


    def train_step(self, data):
        with tf.GradientTape() as tape:
            output, x_true = self(data, training=True)
            t_loss = self.compiled_loss(x_true, output)

        gradients = tape.gradient(t_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(x_true, output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        output, x_true = self(data, training=False)
        t_loss = self.compiled_loss(x_true, output)
        self.compiled_metrics.update_state(x_true, output)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        y_pred, y_true = self(data, training=False)

        rec_pred  = tf.slice(y_pred, [0,2,0], [-1, -1, 1])
        rec_mask  = tf.slice(y_pred, [0,2,1], [-1, -1, 1])
        rec_true  = tf.slice(y_true, [0,1,1], [-1, -1, -1])
        rec_times = tf.slice(y_true, [0,1,0], [-1, -1, 1])
        cls_pred  = tf.argmax(tf.slice(y_pred, [0,0,0], [-1, 2, 1]), 1)
        cls_true  = tf.slice(y_true, [0,0,0], [-1, 1, 1])

        return tf.squeeze(rec_pred), tf.squeeze(rec_mask), \
               tf.squeeze(rec_true), tf.squeeze(rec_times), \
               tf.squeeze(cls_pred), tf.squeeze(cls_true)

    def get_attention(self, data):
        all_vectors = []
        for d in data:
            in_dict = self.input_layer(d)
            enc_output = self.encoder(in_dict, False)
            all_vectors.append(enc_output)

        return all_vectors

    def get_config(self):
        base_config = super(ASTROMER, self).get_config()
        base_config['num_heads'] = self.num_heads
        base_config['num_layers'] = self.num_layers
        base_config['d_model'] = self.d_model
        base_config['num_heads'] = self.num_heads
        base_config['dff'] = self.dff
        base_config['rate'] = self.rate
        base_config['base'] = self.base
        return base_config
