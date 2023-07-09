import tensorflow as tf

def get_probed(input_dict, probed, njobs):

    input_shape = tf.shape(input_dict['input'])

    if probed == 1.:
        probed_mask = tf.ones([input_shape[0], input_shape[1]]) * input_dict['mask']
        input_dict['probed_mask']  = probed_mask
        input_dict['att_mask'] = 1. - probed_mask
        return input_dict

    nprobed = tf.multiply(tf.cast(input_shape[1], tf.float32), probed)
    nprobed = tf.cast(nprobed, tf.int32)
    random_integers = tf.range(input_shape[1], dtype=tf.int32)
    indices = tf.map_fn(lambda x: tf.random.shuffle(random_integers), 
                                      tf.range(input_shape[0]),
                                      parallel_iterations=njobs)
    indices = tf.slice(indices, [0, 0], [-1, nprobed])
    random_mask = tf.one_hot(indices, input_shape[1])
    random_mask = tf.reduce_sum(random_mask, 1)

    input_dict['probed_mask'] = random_mask * input_dict['mask']
    att_mask = (1 - input_dict['mask']) + random_mask 
    att_mask = tf.minimum(att_mask, 1)
    input_dict['att_mask'] = att_mask

    return input_dict