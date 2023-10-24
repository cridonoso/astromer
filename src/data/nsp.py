import tensorflow as tf


def get_segment_length(mask, divide_factor=2):
    window_size = tf.shape(mask)[1]
    n_steps = tf.reduce_sum(mask, 1)
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

def combine_sequences(seq_0, seq_1, mask_0):
    if tf.rank(seq_0) == 3:
        mask_0 = tf.expand_dims(mask_0, axis=-1)

    mask_1 = tf.logical_not(mask_0)

    mask_0 = tf.cast(mask_0, tf.float32)
    mask_1 = tf.cast(mask_1, tf.float32)

    # Make zero all obs to be replace. Keep original observations
    original_masked = seq_0 * mask_0
    
    # Make zero all original observations. Keep randomized ones.
    replace_masked = seq_1 * mask_1

    # Sum both masked_original and masked_raplace to create the final input
    print(original_masked[0])
    print(replace_masked[0])
    randomized_input = original_masked + replace_masked
    print(randomized_input[0])
    return randomized_input


def randomize(input_dict, nsp_prob):
    ''' Mantain times from random light curve by shifting its times'''
    inp_size = tf.shape(input_dict['input_modified'])

    indices = tf.range(0, inp_size[0], dtype=tf.int32)
    shuffle_indices = tf.random.shuffle(indices)

    # Probabilities to randomize
    probs = tf.random.uniform(shape=(inp_size[0],), minval=0, maxval=1)
    binary_vector = tf.where(probs > nsp_prob, 1, 0)
    # On binary_vector = 0, replace the index by a random one
    shuffle_indices = (shuffle_indices*(1-binary_vector)) + (indices*binary_vector)

    # Original vector
    magnitudes_original   = input_dict['input_modified']
    times_original        = tf.slice(input_dict['input'] , [0, 0, 0], [-1, -1, 1])  
    att_mask_original     = input_dict['att_mask']
    prob_mask_original    = input_dict['probed_mask']
    padding_mask_original = input_dict['mask']

    # Candidates for replacing
    magnitudes_replace   = tf.gather(magnitudes_original, shuffle_indices) 
    times_replace        = tf.gather(times_original, shuffle_indices)    
    att_mask_replace     = tf.gather(att_mask_original, shuffle_indices)
    prob_mask_replace    = tf.gather(prob_mask_original, shuffle_indices)
    padding_mask_replace = tf.gather(padding_mask_original, shuffle_indices)

    # Number of observations to be placed as the first original 50% 
    length_0 = get_segment_length(padding_mask_original, divide_factor=2) 

    # Create original fraction mask
    mask_original_obs = creat_mask_given_lenghts(length_0, max_len=inp_size[1])

    # Combine sequences
    # magnitudes_randomized   = combine_sequences(seq_0=magnitudes_original,
    #                                             seq_1=magnitudes_replace, 
    #                                             mask_0=mask_original_obs)
    # times_randomized        = combine_sequences(seq_0=times_original,
    #                                             seq_1=times_replace, 
    #                                             mask_0=mask_original_obs)
    # att_mask_randomized     = combine_sequences(seq_0=att_mask_original,
    #                                             seq_1=att_mask_replace, 
    #                                             mask_0=mask_original_obs)
    # prob_mask_randomized    = combine_sequences(seq_0=prob_mask_original,
    #                                             seq_1=prob_mask_replace, 
    #                                             mask_0=mask_original_obs)
    padding_mask_randomized = combine_sequences(seq_0=padding_mask_original,
                                                seq_1=padding_mask_replace, 
                                                mask_0=mask_original_obs)


    # # Segment embedding 
    # segment_emb = (tf.cast(mask_original_obs, tf.float32)+1.) * padding_mask_randomized
    
    # t_mean = tf.slice(input_dict['mean_values'], [0, 0, 0], [-1, -1, 1])

    # input_dict['nsp_label'] = tf.cast(tf.expand_dims(binary_vector, -1), tf.float32)
    # input_dict['nsp_magnitudes'] = magnitudes_randomized
    # input_dict['nsp_times'] = (times_original + t_mean) * tf.expand_dims(padding_mask_original, axis=-1)
    # input_dict['nsp_pad_mask'] = padding_mask_randomized

    # input_dict['probed_mask'] = prob_mask_randomized
    # input_dict['att_mask'] = att_mask_randomized * (1.-tf.expand_dims(padding_mask_randomized, axis=-1))
    # input_dict['seg_emb'] = segment_emb

    return input_dict

def apply_nsp(dataset, nsp_prob=0.5):

    dataset = dataset.map(lambda x: randomize(x, nsp_prob),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset
