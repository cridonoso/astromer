import tensorflow as tf 


def set_gap_prediction(input_dict, max_gap=0.2):

    inp_size = tf.shape(input_dict['input_modified'])

    max_n_jump = tf.cast(inp_size[0], tf.float32) * max_gap
    max_n_jump = tf.cast(max_n_jump, tf.int32)

    max_n_jump = tf.minimum(max_n_jump, inp_size[0]//3)

    n_jump = tf.random.uniform(shape=(),
                              minval=1,
                              maxval=max_n_jump,
                              dtype=tf.dtypes.int32,
                              name='pivot')


    pivot = tf.random.uniform(shape=(),
                              minval=inp_size[0]//3,
                              maxval=inp_size[0]-inp_size[0]//3,
                              dtype=tf.dtypes.int32,
                              name='pivot')
    
    times = tf.slice(input_dict['input'], 
                     [0, 0], 
                     [-1, 1], 
                     name='times')
    input_dict['t0']  = tf.slice(times, [pivot-1, 0], [1, 1])
    input_dict['t0']  = tf.squeeze(input_dict['t0'])
    input_dict['t1']  = tf.slice(times, [pivot+n_jump, 0], [1, 1])
    input_dict['t1']  = tf.squeeze(input_dict['t1'])
    input_dict['dt'] = input_dict['t1']  - input_dict['t0']

    gap_mask = tf.concat([
            tf.ones([pivot, 1], dtype=tf.float32),
            tf.zeros([n_jump, 1], dtype=tf.float32),
            tf.ones([inp_size[0]-(pivot+n_jump), 1], dtype=tf.float32),
        ], axis=0)

    input_dict['gap_mask'] = gap_mask
    input_dict['att_mask'] = 1. - input_dict['att_mask']


    seg_emb = tf.concat([
            tf.ones([pivot, 1], dtype=tf.float32)*-1.,
            tf.zeros([n_jump, 1], dtype=tf.float32),
            tf.ones([inp_size[0]-(pivot+n_jump), 1], dtype=tf.float32),
        ], axis=0)


    input_dict['seg_emb'] = seg_emb * tf.expand_dims(input_dict['mask'], axis=-1)

    return input_dict

def set_gap(dataset, max_gap=0.2):

    dataset = dataset.map(lambda x: set_gap_prediction(x, max_gap=max_gap),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

def invert_mask(x_inp, y_inp):
    x_inp['att_mask'] = 1. - x_inp['att_mask']
    return x_inp, y_inp