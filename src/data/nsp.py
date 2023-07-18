import tensorflow as tf

from .preprocessing import standardize_batch

def get_mask(steps, max_len):
    steps= tf.cast(steps, tf.int32)
    half = tf.cast(tf.divide(steps, 2), tf.int32)
    mask_0 = tf.zeros([half], dtype=tf.float32)
    mask_1 = tf.ones([steps-half], dtype=tf.float32)
    mask_2 = tf.zeros([max_len-steps], dtype=tf.float32)
    return tf.concat([mask_0, mask_1, mask_2], 0)
    
def randomize(input_dict, nsp_prob):
    n_steps = tf.reduce_sum(input_dict['mask'], 1)
    inp_size = tf.shape(input_dict['input'])

    replace = tf.random.shuffle(input_dict['input'])
    replace = tf.reverse(replace, axis=[1])

    input_dict['input'] = standardize_batch(input_dict['input'])
    replace = standardize_batch(replace)

    # will be one on the part that we are going to replace
    mask = tf.map_fn(lambda x: get_mask(x, max_len=inp_size[1]), n_steps)

    probs = tf.random.uniform(shape=(inp_size[0],), minval=0, maxval=1)
    binary_vector = tf.where(probs < nsp_prob, 1., 0.)
    binary_vector = tf.expand_dims(binary_vector, -1)

    mask_replace  = binary_vector*mask
    mask_replace = tf.expand_dims(mask_replace, -1)
    mask_preserve = 1.-mask_replace

    input_dict['a'] =  mask_replace
    input_dict['b'] =  mask_preserve
    
    replace = replace * mask_replace
    original = input_dict['input']  * mask_preserve

    padding_mask = tf.expand_dims(input_dict['mask'], -1)
    input_dict['nsp_label'] = 1.-binary_vector
    input_dict['nsp_input'] = replace*mask_replace + original*mask_preserve*padding_mask
    input_dict['seg_emb'] = mask

    return input_dict
