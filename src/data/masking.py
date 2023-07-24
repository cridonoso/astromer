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


def add_random(input_dict, random_frac, njobs):
    """ Add random observations to each sequence
        
    Args:
        random_frac (number): Fraction of probed (in decimal) to be replaced with random values
    """ 
    input_shape = tf.shape(input_dict['input'])
    input_dict['input_pre_nsp'] = input_dict['input'] 

    n_probed = tf.reduce_sum(input_dict['probed_mask'], 1)
    n_random = tf.math.ceil(n_probed * random_frac)
    n_random = tf.cast(n_random, tf.int32)

    # ====== RANDOM MASK =====
    random_mask = tf.map_fn(lambda x: tf.one_hot(
                                                tf.slice(
                                                    tf.random.shuffle(
                                                        tf.where(x[0])
                                                        ), 
                                                    [0, 0], [x[1], 1]),
                                            input_shape[1], dtype=tf.int32), 
                                  (tf.cast(input_dict['probed_mask'], tf.int32), n_random),
                                  parallel_iterations=njobs,
                                  fn_output_signature=tf.int32)
    random_mask = tf.reduce_sum(random_mask, 1)
    random_mask = tf.reshape(random_mask, [-1, input_shape[1]])

    rest = tf.cast(input_dict['probed_mask'], tf.int32) * (1-random_mask)

    # ====== SAME MASK =====
    same_mask = tf.map_fn(lambda x: tf.one_hot(
                                                tf.slice(
                                                    tf.random.shuffle(
                                                        tf.where(x[0])
                                                        ), 
                                                    [0, 0], [x[1], -1]),
                                            input_shape[1], dtype=tf.int32), 
                                  (tf.cast(rest, tf.int32), n_random),
                                  parallel_iterations=njobs,
                                  fn_output_signature=tf.int32)
    same_mask = tf.reduce_sum(same_mask, 1)
    same_mask = tf.reshape(same_mask, [-1, input_shape[1]])
    # ===== REPLACEMENT ==== 
    random_replacement = tf.random.shuffle(tf.transpose(input_dict['input'], [1, 0, 2]))
    random_replacement = tf.transpose(random_replacement, [1, 0, 2])
    random_replacement = random_replacement * tf.cast(tf.expand_dims(random_mask, -1), tf.float32) * [0., 1., 1.]

    # Mask refering to observations that do not change
    keep_mask = tf.expand_dims(1 - random_mask, -1)
    keep_mask = tf.tile(keep_mask, [1, 1, input_shape[-1]-1])
    keep_mask = tf.concat([tf.zeros([input_shape[0], input_shape[1], 1], dtype=tf.int32), keep_mask], 2)
    keep_mask = tf.abs([1, 0, 0] - keep_mask)

    # Part of the input we mantain
    keep_input = input_dict['input'] * tf.cast(keep_mask, tf.float32)

    # Replacing original input with the randomized one
    input_dict['input']  = random_replacement + keep_input

    # Attention mask is 1 when masked. 
    # Random mask is 1 for masked observations selected to be randomized
    # then,
    input_dict['att_mask'] = input_dict['att_mask'] * tf.cast(1 - random_mask, tf.float32)

    # We do the same but this time, unmasking observations to be visible during the attention mechanism
    input_dict['att_mask'] = input_dict['att_mask'] + tf.cast(same_mask, tf.float32)

    return input_dict