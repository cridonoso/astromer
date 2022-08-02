import tensorflow as tf

@tf.function
def reshape_mask(mask):
    ''' Reshape Mask to match attention dimensionality '''
    with tf.name_scope("reshape_mask") as scope:
        return mask[:, tf.newaxis, tf.newaxis, :, 0]

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
    orig_magn = seq_magn

    # [MASK] values
    mask_out = get_masked(seq_magn, msk_frac)

    # [MASK] -> Same values
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_out,
                                   seq_magn,
                                   same_frac,
                                   name='set_same')

    # [MASK] -> Random value
    seq_magn, mask_in = set_random(seq_magn,
                                   mask_in,
                                   tf.random.shuffle(seq_magn),
                                   rnd_frac,
                                   name='set_random')

    time_steps = tf.shape(seq_magn)[0]

    mask_out = tf.reshape(mask_out, [time_steps, 1])
    mask_in = tf.reshape(mask_in, [time_steps, 1])

    if time_steps < max_obs:
        mask_fill = tf.ones([max_obs - time_steps, 1], dtype=tf.float32)
        mask_out  = tf.concat([mask_out, 1-mask_fill], 0)
        mask_in   = tf.concat([mask_in, mask_fill], 0)
        seq_magn   = tf.concat([seq_magn, 1-mask_fill], 0)
        seq_time   = tf.concat([seq_time, 1-mask_fill], 0)
        orig_magn   = tf.concat([orig_magn, 1-mask_fill], 0)

    input_dict['input_modified'] = seq_magn
    input_dict['mask_in']  = mask_in
    input_dict['mask_out'] = mask_out

    return input_dict


def mask_batch(batch, msk_frac, rnd_frac, same_frac):
    inputs = batch['input']

    input_shape = tf.shape(batch['input'])

    indices = tf.map_fn(lambda x: tf.random.shuffle(
                                    tf.range(input_shape[1], dtype=tf.float32)),
                        elems=(batch['input']))
    # N for masking
    n = tf.cast(
            tf.cast(input_shape[1], tf.float32)*msk_frac,
            dtype=tf.int32, name='num_to_mask')

    # indices = tf.slice(indices.stack(), [0,0],[-1, n])
    indices = tf.stack(indices, 0)
    indices = tf.slice(indices, [0,0],[-1, n])
    indices = tf.cast(indices, tf.int32)

    # Creates attention mask
    mask_in = tf.one_hot(indices, input_shape[1])
    mask_in = tf.reduce_sum(mask_in, 1)
    mask_out = tf.minimum(mask_in, 1)

    # N for random and same
    n_rnd = tf.cast(
                tf.cast(input_shape[1], tf.float32)*rnd_frac,
                dtype=tf.int32, name='mask_to_rand')
    n_sme = tf.cast(
                tf.cast(input_shape[1], tf.float32)*same_frac,
                dtype=tf.int32, name='mask_to_same')

    # From all the indices we select n_rnd and n_sme to replace
    rand_inds = tf.slice(indices, [0,0],[-1, n_rnd])
    same_inds = tf.slice(indices, [0,n_rnd],[-1, n_sme])

    # Creates same/random binary masks
    same_mask = tf.one_hot(same_inds, input_shape[1])
    same_mask = tf.reduce_sum(same_mask, 1)

    rnd_mask = tf.one_hot(rand_inds, input_shape[1])
    rnd_mask = tf.reduce_sum(rnd_mask, 1)

    # Change values in original mask to be visible
    mask_in_rnd_sme = tf.minimum(same_mask + rnd_mask, 1.)
    mask_in_rnd_sme = tf.minimum(mask_out, 1.-mask_in_rnd_sme)

    # We take only what we need (FOR FUTURE WORKS ON MULTIBAND CHANGE THIS)
    magnitudes = tf.slice(batch['input'], [0,0,1], [-1,-1, 1])

    # Choose random values and replace masked ones
    rnd_elem = tf.random.normal(tf.shape(rnd_mask), 0, tf.math.reduce_std(magnitudes, 1))
    rnd_elem = tf.reshape(rnd_elem, [tf.shape(rnd_elem)[0], tf.shape(rnd_elem)[1]])
    rnd_elem = tf.multiply(rnd_elem, rnd_mask)

    # Reeplace random values in the magnitudes
    magnitudes_copy = tf.reshape(magnitudes, [input_shape[0], input_shape[1]])
    magnitudes_copy = tf.multiply(magnitudes_copy, 1.-rnd_mask)
    magnitudes_copy = magnitudes_copy + rnd_elem

    # intersection with padding mask
    padd_mask = tf.cast(batch['mask'], tf.float32)
    mask_out = tf.multiply(padd_mask, mask_out)
    mask_in_rnd_sme = tf.maximum(1.-padd_mask, mask_in_rnd_sme)

    # Concat the randomized input with real times
    batch['input_modified'] = tf.expand_dims(magnitudes_copy, 2)
    batch['mask_in']  = tf.expand_dims(mask_in_rnd_sme, 2)
    batch['mask_out'] = tf.expand_dims(mask_out, 2)

    return batch


def mask_dataset(dataset,
                 msk_frac=.5,
                 rnd_frac=.2,
                 same_frac=.2,
                 per_sample_mask=False,
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
    if per_sample_mask:
        assert window_size is not None, 'Masking per sample needs window_size to be specified'
        dataset = dataset.map(lambda x: mask_sample(x,
                                                    msk_frac=msk_frac,
                                                    rnd_frac=rnd_frac,
                                                    same_frac=same_frac,
                                                    max_obs=window_size),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(lambda x: mask_batch(x,
                                                   msk_frac=msk_frac,
                                                   rnd_frac=rnd_frac,
                                                   same_frac=same_frac),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset
