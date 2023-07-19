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

def get_segment_length(mask, window_size, divide_factor=2):
    n_steps = tf.reduce_sum(mask, 1)
    half_current = tf.math.divide(n_steps, divide_factor)
    half_maximum = tf.math.divide(window_size, divide_factor)
    length = tf.minimum(tf.cast(n_steps, tf.float32), 
                        tf.cast(half_maximum, tf.float32))
    return tf.cast(length, tf.int32)

def creat_mask_given_lenghts(lengths, max_len):
    valid_ind = tf.ragged.range(0, tf.cast(lengths, tf.int32))
    mask = tf.one_hot(valid_ind, max_len)
    mask = tf.reduce_sum(mask, 1)
    return tf.cast(mask, tf.bool)

def stichfix(tensor, pos=0):

    vector = tf.slice(tensor, [0, 0, pos], [-1, -1, 1])


    # tomar el maximo
    # normalizar el segundo trozo entre 0 y 1 
    # sumarle el maximo + mean_cadence
    # para eso necesitamos saber donde esta la mitad
    # quizas deberia hacerse antes 



def concat_segments(segment_0, segment_1, mask_0, mask_1):
    sub_0 = tf.ragged.boolean_mask(segment_0, mask_0)
    sub_1 = tf.ragged.boolean_mask(segment_1, mask_1)        
    return tf.concat([sub_0, sub_1], axis=1).to_tensor()

def randomize_v2(input_dict, nsp_prob):
    inp_size = tf.shape(input_dict['input'])

    indices = tf.range(0, inp_size[0], dtype=tf.int32)
    indices = tf.random.shuffle(indices)

    to_replace = tf.gather(input_dict['input'], indices)

    # standardize magnitudes to have zero-mean
    input_dict['input'] = standardize_batch(input_dict['input'])
    replace             = standardize_batch(replace)

    # get number of observations to be part of each segment 
    length_0 = get_segment_length(input_dict['mask'], inp_size[1]) 
    length_1 = get_segment_length(input_dict['mask'], inp_size[1])

    # cut sequence



    probs = tf.random.uniform(shape=(inp_size[0],), minval=0, maxval=1)
    binary_vector = tf.where(probs < nsp_prob, 1., 0.)
    binary_vector = tf.expand_dims(binary_vector, -1)

    mask_replace  = binary_vector*mask
    mask_replace = tf.expand_dims(mask_replace, -1)
    mask_preserve = 1.-mask_replace

    
    replace = replace * mask_replace

    original = input_dict['input']  * mask_preserve

    padding_mask = tf.expand_dims(input_dict['mask'], -1)
    input_dict['nsp_label'] = 1.-binary_vector
    input_dict['nsp_input'] = replace*mask_replace + original*mask_preserve*padding_mask
    input_dict['seg_emb'] = mask

    return input_dict