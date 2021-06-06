import tensorflow as tf
import json
import os, sys
sys.path.append('./astromer')

from core.astromer import get_ASTROMER


def standardize_mag(tensor):
    mean_value = tf.reduce_mean(tensor, 0, name='mean_value')
    std_value = tf.math.reduce_std(tensor, 0, name='std_value')

    normed = tf.where(std_value == 0.,
                     (tensor - mean_value),
                     (tensor - mean_value)/std_value)
    return normed


class BASE_ASTROMER(object):
    """docstring for BASE_ASTROMER. Only Magnitudes"""
    def __init__(self, weigths='./astromer/runs/macho_old_pe/finetuning/model_1'):
        super(BASE_ASTROMER, self).__init__()
        self.weights_path = weigths

        conf_file = os.path.join(weigths, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)

        model = get_ASTROMER(num_layers=conf['layers'],
                                  d_model   =conf['head_dim'],
                                  num_heads =conf['heads'],
                                  dff       =conf['dff'],
                                  base      =conf['base'],
                                  dropout   =conf['dropout'],
                                  maxlen    =conf['max_obs'])

        print('[INFO] Loading ASTROMER Embedding...')
        weights_path = '{}/weights'.format(self.weights_path)
        model.load_weights(weights_path)
        self.encoder = model.get_layer('encoder')

        self.steps = self.encoder.input['input'].shape[1] - 2
        self.inp_dim = self.encoder.input['input'].shape[-1]

    def __call__(self, inputs, times):
        inputs = tf.concat([inputs, times], 1)
        length = tf.shape(inputs)[0]
        rest = length % self.steps
        filler = tf.zeros([self.steps-rest, 2])

        fullvec = tf.concat([inputs, filler], 0)
        clstkn = tf.tile(tf.cast([[-99.]], tf.float32),
                         [1, self.inp_dim], name='cls_tkn')
        septkn = tf.tile(tf.cast([[-98.]], tf.float32),
                         [1, self.inp_dim], name='sep_tkn')
        msktkn = tf.zeros([1], name='msk_tkn')
        rangex = tf.range(0, tf.shape(fullvec)[0], self.steps,
                  dtype=tf.float32)


        def fn(index):
            mask_1 = tf.zeros(self.steps)
            single = tf.slice(fullvec, [tf.cast(index, tf.int32),0],[self.steps, -1])

            times = tf.slice(single, [0,0],[-1,1])
            value = tf.slice(single, [0,1],[-1,1])

            serie_1 = standardize_mag(value)

            if index == rangex[-1]:
                mask_1 = tf.sequence_mask(rest, self.steps)
                mask_1 = tf.logical_not(mask_1)
                mask_1 = tf.cast(mask_1, tf.float32)

            times = tf.concat([clstkn, times, septkn], 0, name='times')
            times = tf.expand_dims(times, 0)
            serie = tf.concat([clstkn, serie_1, septkn], 0, name='input')
            serie = tf.expand_dims(serie, 0)
            mask  = tf.concat([msktkn, mask_1, msktkn], 0, name='mask')
            mask = tf.expand_dims(tf.expand_dims(mask, 1), 0)

            inp_data = {'mask': mask, 'times':times, 'input':serie}
            x = self.encoder(inp_data)

            x = tf.slice(x, [0, 1, 0], [-1, tf.shape(x)[1]-2, -1])
            mask = tf.slice(mask, [0, 1, 0], [-1, tf.shape(mask)[1]-2, -1])

            return tf.squeeze(tf.concat([x, mask], 2))

        response = tf.map_fn(lambda x: fn(x), rangex)

        dim_tensor = tf.shape(response)
        response = tf.reshape(response,
                              [dim_tensor[0]*dim_tensor[1], dim_tensor[2]])

        params = tf.shape(response)[-1]
        values = tf.slice(response, [0, 0], [-1, params-1])
        mask  = tf.slice(response, [0, params-1], [-1, 1])
        mask = tf.logical_not(tf.cast(mask, tf.bool))

        mask = tf.tile(mask, [1, params-1])
        emb = tf.boolean_mask(values, mask)
        emb = tf.reshape(emb, [tf.shape(emb)[0]/(params-1), params-1])
        return emb
