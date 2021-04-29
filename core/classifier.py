import tensorflow as tf
import json

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

from core.input   import InputLayer
from core.output  import OutputLayer
from core.encoder import Encoder
from core.decoder import Decoder

from core.transformer import ASTROMER

class ASTROMERClassifier(Model):
    def __init__(self, config, **kwargs):

        super().__init__(name='Classifier')

        with open(config, 'r') as handle:
            conf = json.load(handle)

        self.astromer = ASTROMER(num_layers=conf['layers'],
                                 d_model=conf['head_dim'],
                                 num_heads=conf['heads'],
                                 dff=conf['dff'],
                                 rate=conf['dropout'],
                                 base=conf['base'],
                                 **kwargs)

        expdir = '{}/train_model.h5'.format(conf['p'])

        # New Layers
        self.cls_layer = Dense(5, name='CLFLayer')


    def call(self, inputs, training=False):
        # Join series by [SEP]
        in_dict = self.astromer.input_layer(inputs)
        enc_output = self.astromer.encoder(in_dict, training)

        logist_rec = tf.slice(enc_output, [0,1,0], [-1, -1, -1],
        					  name='RecontructionSplit')
        logist_cls = tf.slice(enc_output, [0,0,0], [-1, 1, -1],
        					  name='ClassPredictedSplit')
        # CLASSIFICATION
        cls_prob = self.cls_layer(logist_cls)
        cls_prob = tf.transpose(cls_prob, [0,2,1],
        						name='CategoricalClsPred')

        # RECONSTRUCTION
        reconstruction = self.astromer.output_layer.reg_layer(logist_rec)

        print(cls_prob)
        print(reconstruction)

        # final_output = tf.concat([cls_prob, reconstruction],
        #                          axis=1,
        #                          name='ConcatClassRec')
        #
        # # we need to adjust our mask to match the current dimensionality
        # in_dict['tar_mask'] = tf.concat([tf.expand_dims(in_dict['tar_mask'][:, 0], 1),
        #                                  in_dict['tar_mask']], 1, name='RepeatClassMask')
        #
        # output_mask = tf.concat([final_output, tf.expand_dims(in_dict['tar_mask'], 2)],
        #                         2,
        #                         name='ConcatPredsAndMask')

        return reconstruction, in_dict['target']

    def model(self, batch_size):
        serie_1  = Input(shape=(50, 3), batch_size=batch_size, name='Serie1')
        serie_2  = Input(shape=(50, 3), batch_size=batch_size, name='Serie2')

        steps_1 = Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='steps_1')
        steps_2 = Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='steps_2')

        data = {'serie_1':serie_1, 'serie_2':serie_2,
                'steps_1':steps_1, 'steps_2':steps_2}

        self.astromer.build(data)

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

    def get_model():
        return ASTROMERClassifier(name='ASTROMER')
