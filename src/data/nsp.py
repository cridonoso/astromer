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

def creat_mask_given_lenghts(lengths_1, max_len):
    valid_ind = tf.ragged.range(0, tf.cast(lengths_1, tf.int32))
    mask = tf.one_hot(valid_ind, max_len)
    mask = tf.reduce_sum(mask, 1)
    return tf.cast(mask, tf.bool)

def concat_segments(segment_0, segment_1, mask_0, mask_1, padding_mask=None):
    sub_0 = tf.ragged.boolean_mask(segment_0, mask_0)

    if padding_mask is not None:
        time_1 = tf.slice(segment_1, [0,0,0], [-1, -1, 1])
        rest   = tf.slice(segment_1, [0,0,1], [-1, -1, -1])

        valid_values = tf.ragged.boolean_mask(time_1, tf.cast(padding_mask, tf.bool))
        min_val = tf.reduce_min(valid_values, axis=1) 
        min_val = tf.expand_dims(min_val, axis=1)
        max_val = tf.reduce_max(valid_values, axis=1) 
        max_val = tf.expand_dims(max_val, axis=1)
        time_1 = tf.math.divide_no_nan(time_1 - min_val, max_val-min_val)
        
        last_time_0 = tf.reduce_max(sub_0, axis=1)
        last_time_0 = tf.slice(last_time_0, [0, 0], [-1, 1])
        last_time_0 = tf.reshape(last_time_0, [tf.shape(last_time_0)[0], 1, 1])
        time_0_a = tf.slice(segment_0, [0, 0, 0], [-1, tf.shape(segment_0)[1]-1, 1])
        time_0_b = tf.slice(segment_0, [0, 1, 0], [-1, -1, 1])
        delta_t0 = time_0_b - time_0_a
        delta_t0 = tf.ragged.boolean_mask(delta_t0, tf.slice(mask_0, [0, 1], [-1, -1]))
        delta_t0 = tf.reduce_mean(delta_t0, 1)
        delta_t0 = tf.reshape(delta_t0, [tf.shape(delta_t0)[0], 1, 1])
        time_1 = time_1 + delta_t0 + last_time_0
        segment_1 = tf.concat([time_1, rest], 2)

    sub_1 = tf.ragged.boolean_mask(segment_1, mask_1)   
     
    return tf.concat([sub_0, sub_1], axis=1).to_tensor()

def randomize_v2(input_dict, nsp_prob):
    inp_size = tf.shape(input_dict['input'])

    indices = tf.range(0, inp_size[0], dtype=tf.int32)
    shuffle_indices = tf.random.shuffle(indices)

    # Probabilities to randomize
    probs = tf.random.uniform(shape=(inp_size[0],), minval=0, maxval=1)
    binary_vector = tf.where(probs < nsp_prob, 1, 0)
    shuffle_indices = (shuffle_indices*(1-binary_vector)) + (indices*binary_vector)

    # Candidates for replacing
    x_replace = tf.gather(input_dict['input'], shuffle_indices)
    att_mask_replace = tf.gather(input_dict['att_mask'], shuffle_indices)
    prob_mask_replace = tf.gather(input_dict['probed_mask'], shuffle_indices)
    padding_mask_replace = tf.gather(input_dict['mask'], shuffle_indices)

    # standardize magnitudes to have zero-mean
    input_dict['input'] = standardize_batch(input_dict['input'])
    x_replace           = standardize_batch(x_replace)

    # get number of observations to be part of each segment 
    length_0 = get_segment_length(input_dict['mask'], inp_size[1]) 
    length_1 = inp_size[1] - length_0

    # create mask to extract values 
    mask_0 = creat_mask_given_lenghts(length_0, inp_size[1])
    mask_0_r = tf.logical_not(mask_0)
    mask_0_r = tf.cast(mask_0_r, tf.int32) * tf.expand_dims(binary_vector, -1)

    mask_1 = creat_mask_given_lenghts(length_1, inp_size[1])
    mask_1 = tf.cast(mask_1, tf.int32) * tf.expand_dims(1-binary_vector, -1)
    mask_1 = mask_1 + mask_0_r
    mask_1 = tf.cast(mask_1, tf.bool)

    # concat segments
    nsp_input   = concat_segments(input_dict['input'], x_replace, mask_0, mask_1, padding_mask=padding_mask_replace)
    att_mask    = concat_segments(input_dict['att_mask'], att_mask_replace, mask_0, mask_1)
    probed_mask = concat_segments(input_dict['probed_mask'], prob_mask_replace, mask_0, mask_1)
    probed_mask = concat_segments(input_dict['mask'], padding_mask_replace, mask_0, mask_1)

    # Segment embedding 
    segment_emb = (tf.cast(mask_0, tf.float32)+1.) * input_dict['mask']

    input_dict['mask'] = padding_mask_replace * input_dict['mask']
    input_dict['nsp_label'] = tf.cast(tf.expand_dims(binary_vector, -1), tf.float32)
    input_dict['nsp_input'] = nsp_input
    input_dict['nsp_pad_mask'] = probed_mask

    input_dict['probed_mask'] = probed_mask
    input_dict['att_mask'] = att_mask
    input_dict['seg_emb'] = segment_emb

    return input_dict