import tensorflow as tf



@tf.function
def get_masked(tensor, frac=0.15):
    """ Add [MASK] values to be predicted
    Args:
        tensor : tensor values
        frac (float, optional): percentage for masking [MASK]
    Returns:
        binary tensor: a time-distributed mask
    """
    with tf.name_scope("get_masked") as scope:
        steps = tf.shape(tensor)[0] # time steps
        nmask = tf.multiply(tf.cast(steps, tf.float32), frac)
        nmask = tf.cast(nmask, tf.int32, name='nmask')

        indices = tf.range(steps)
        indices = tf.random.shuffle(indices)
        indices = tf.slice(indices, [0], [nmask])

        mask = tf.reduce_sum(tf.one_hot(indices, steps), 0)
        mask = tf.minimum(mask, tf.ones_like(mask))
        return mask

@tf.function
def set_random(serie_1, mask_1, serie_2, rnd_frac, name='set_random'):
    """ Add Random values in serie_1
    Note that if serie_2 == serie_1 then it replaces the true value
    Args:
        serie_1: current serie
        mask_1 : mask containing the [MASKED]-indices from serie_1
        serie_2: random values to be placed within serie_1
        rnd_frac (float): fraction of [MASKED] to be replaced by random
                          elements from serie_2
    Returns:
        serie_1: serie_1 with random values
    """
    with tf.name_scope(name) as scope:
        nmasked = tf.reduce_sum(mask_1)
        nrandom = tf.multiply(nmasked, rnd_frac, name='mulscalar')
        nrandom = tf.cast(tf.math.ceil(nrandom), tf.int32)

        mask_indices = tf.where(mask_1)
        mask_indices = tf.random.shuffle(mask_indices)
        mask_indices = tf.reshape(mask_indices, [-1])
        mask_indices = tf.slice(mask_indices, [0], [nrandom])

        rand_mask = tf.one_hot(mask_indices, tf.shape(mask_1)[0])
        rand_mask = tf.reduce_sum(rand_mask, 0)
        rand_mask = tf.minimum(rand_mask, tf.ones_like(rand_mask))
        rand_mask = tf.expand_dims(rand_mask, 1)
        rand_mask = tf.tile(rand_mask, [1, tf.shape(serie_2)[-1]])
        
        len_s1 = tf.minimum(tf.shape(serie_2)[0],
                            tf.shape(rand_mask)[0])

        serie_2 = tf.slice(serie_2, [0,0], [len_s1, -1])

        rand_vals = tf.multiply(serie_2, rand_mask, name='randvalsmul')

        keep_mask = tf.math.floor(tf.math.cos(rand_mask))

        serie_1 = tf.multiply(serie_1, keep_mask, name='seriemul')

        keep_mask = tf.slice(keep_mask, [0,0], [-1,1])
        mask_1  = tf.multiply(mask_1, tf.squeeze(keep_mask), name='maskmul2')
        serie_1 = tf.add(serie_1, rand_vals)

        return serie_1, mask_1

def mask_sample(input_dict, msk_frac, rnd_frac, same_frac, max_obs):
    '''
    OLD VERSION
    '''
    x = input_dict['input']

    seq_time = tf.slice(x, [0, 0], [-1, 1])
    seq_magn = tf.slice(x, [0, 1], [-1, 1])
    seq_errs = tf.slice(x, [0, 2], [-1, 1])

    # Save the true values
    time_steps = tf.shape(seq_magn)[0]
    orig_magn = seq_magn

    # [MASK] values
    if msk_frac == 0.:
        mask_out = tf.ones(time_steps)
    else:
        mask_out = get_masked(seq_magn, msk_frac)

    # [MASK] -> Identity
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_out,
                                   seq_magn,
                                   same_frac,
                                   name='set_same')

    # [MASK] -> Random
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_in,
                                   tf.random.shuffle(seq_magn),
                                   rnd_frac,
                                   name='set_random')
    if msk_frac == 0.:
        mask_in  =  1.- mask_out

    mask_out = tf.reshape(mask_out, [time_steps, 1])
    mask_in = tf.reshape(mask_in, [time_steps, 1])

    if time_steps < max_obs:
        mask_fill = tf.ones([max_obs - time_steps, 1], dtype=tf.float32)
        mask_out  = tf.concat([mask_out,  1-mask_fill], 0)
        mask_in   = tf.concat([mask_in,     mask_fill], 0)
        seq_magn  = tf.concat([seq_magn,  1-mask_fill], 0)
        seq_time  = tf.concat([seq_time,  1-mask_fill], 0)
        orig_magn = tf.concat([orig_magn, 1-mask_fill], 0)
        input_dict['mask'] =  tf.concat([input_dict['mask'],
                                        1-tf.reshape(mask_fill, [tf.shape(mask_fill)[0]])], 0)

        reshaped_mask = tf.zeros([max_obs - time_steps,
                                  tf.shape(input_dict['input'])[-1]],
                                  dtype=tf.float32)

        input_dict['input'] = tf.concat([input_dict['input'], reshaped_mask], 0)

    input_dict['input_modified'] = seq_magn
    input_dict['att_mask']       = mask_in
    input_dict['probed_mask']    = mask_out

    return input_dict

def mask_dataset(dataset,
                 msk_frac=.5,
                 rnd_frac=.2,
                 same_frac=.2,
                 window_size=None):
    """
    Mask samples per batch following BERT strategy

    Args:
        dataset: A batched tf.Dataset
        msk_frac: observations fraction per light curve that will be masked
        rnd_frac: fraction from masked values to be replaced by random values
        same_frac: fraction from masked values to be replace by same values

    Returns:
        type: tf.Dataset
    """
    assert window_size is not None, 'Masking per sample needs window_size to be specified'
    dataset = dataset.map(lambda x: mask_sample(x,
                                                msk_frac=msk_frac,
                                                rnd_frac=rnd_frac,
                                                same_frac=same_frac,
                                                max_obs=window_size),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    shapes = {'input' :[None, 3],
              'lcid'  :(),
              'length':(),
              'mask'  :[None, ],
              'label' :(),
              'input_modified': [None, None],
              'att_mask': [None, None],
              'probed_mask': [None, None],
              'mean_values':[None, None]}

    return dataset, shapes

# ======================================
# ======================================
# ======================================
# ======================================
# ======================================

# def get_probed(input_dict, probed, njobs):

#     input_shape = tf.shape(input_dict['input']) # (batch x steps x 3)

#     if probed == 1.:
#         probed_mask = tf.ones([input_shape[0], input_shape[1]]) * input_dict['mask']
#         input_dict['probed_mask']  = probed_mask
#         input_dict['att_mask'] = 1. - probed_mask
#         return input_dict
    
    
#     nprobed = tf.multiply(tf.cast(input_shape[1], tf.float32), probed)
#     nprobed = tf.cast(nprobed, tf.int32)
#     random_integers = tf.range(input_shape[1], dtype=tf.int32)
    
#     indices = tf.map_fn(lambda x: tf.random.shuffle(random_integers), 
#                                       tf.range(input_shape[0]),
#                                       parallel_iterations=njobs)
#     indices = tf.slice(indices, [0, 0], [-1, nprobed])
#     random_mask = tf.one_hot(indices, input_shape[1])
#     random_mask = tf.reduce_sum(random_mask, 1)
    
#     input_dict['probed_mask'] = random_mask*input_dict['mask']
    
#     att_mask = (1 - input_dict['mask']) + random_mask 
#     att_mask = tf.minimum(att_mask, 1)
#     input_dict['att_mask'] = att_mask

#     return input_dict

# def create_mask(pre_mask, n_elements):
#     indices = tf.where(pre_mask)
#     indices = tf.random.shuffle(indices)
#     indices = tf.slice(indices, [0, 0], [n_elements, -1])

#     mask = tf.one_hot(indices, tf.shape(pre_mask)[0], dtype=tf.int32)
#     mask = tf.reduce_sum(mask, 0)
#     mask = tf.reshape(mask, [tf.shape(pre_mask)[0]])
    
#     return mask

# def add_random(input_dict, random_frac, njobs):
#     """ Add random observations to each sequence
        
#     Args:
#         random_frac (number): Fraction of probed (in decimal) to be replaced with random values
#     """ 
#     input_shape = tf.shape(input_dict['input'])
#     input_dict['input_pre_nsp'] = input_dict['input'] 

#     # ====== RANDOM MASK =====
#     n_probed = tf.reduce_sum(input_dict['probed_mask'], 1)
#     n_random = tf.math.ceil(n_probed * random_frac)
#     n_random = tf.cast(n_random, tf.int32)
#     random_mask = tf.map_fn(lambda x: create_mask(x[0], x[1]),
#                                 (input_dict['probed_mask'], n_random),
#                                 parallel_iterations=njobs,
#                                 fn_output_signature=tf.int32)

#     # ====== SAME MASK =====
#     rest = tf.cast(input_dict['probed_mask'], tf.int32) * (1-random_mask)
#     n_rest = tf.reduce_sum(rest, 1)
#     n_same = tf.math.ceil(tf.cast(n_rest, tf.float32) * random_frac)
#     n_same = tf.cast(n_same, tf.int32)

#     same_mask = tf.map_fn(lambda x: create_mask(x[0], x[1]), 
#                                   (rest, n_same),
#                                   parallel_iterations=njobs,
#                                   fn_output_signature=tf.int32)

#     # ===== REPLACEMENT ==== 
#     random_replacement = tf.random.shuffle(tf.transpose(input_dict['input'], [1, 0, 2]))
#     random_replacement = tf.transpose(random_replacement, [1, 0, 2])
#     random_replacement = random_replacement * tf.cast(tf.expand_dims(random_mask, -1), tf.float32) * [0., 1., 1.]

#     # Mask refering to observations that do not change
#     keep_mask = tf.expand_dims(1 - random_mask, -1)
#     keep_mask = tf.tile(keep_mask, [1, 1, input_shape[-1]-1])
#     keep_mask = tf.concat([tf.zeros([input_shape[0], input_shape[1], 1], dtype=tf.int32), keep_mask], 2)
#     keep_mask = tf.abs([1, 0, 0] - keep_mask)

#     # Part of the input we mantain
#     keep_input = input_dict['input'] * tf.cast(keep_mask, tf.float32)

#     # Replacing original input with the randomized one
#     input_dict['input']  = random_replacement + keep_input

#     # Attention mask is 1 when masked. 
#     # Random mask is 1 for masked observations selected to be randomized
#     # then,
#     att_mask    = tf.cast(input_dict['att_mask'], tf.bool)
#     random_mask = tf.cast(random_mask, tf.bool)
#     same_mask   = tf.cast(same_mask, tf.bool)

#     att_mask = tf.math.logical_xor(att_mask, random_mask)
#     att_mask = tf.math.logical_xor(att_mask, same_mask)

#     input_dict['att_mask'] = tf.cast(att_mask, tf.float32)

#     return input_dict