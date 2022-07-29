import tensorflow as tf

@tf.function
def reshape_mask(mask):
    ''' Reshape Mask to match attention dimensionality '''
    with tf.name_scope("reshape_mask") as scope:
        return mask[:, tf.newaxis, tf.newaxis, :, 0]

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
    same_inds = tf.slice(indices, [0,0],[-1, n_sme])
    rand_inds = tf.slice(indices, [0,n_sme],[-1, n_rnd])

    # Creates same/random binary masks
    same_mask = tf.one_hot(same_inds, input_shape[1])
    same_mask = tf.reduce_sum(same_mask, 1)

    rnd_mask = tf.one_hot(rand_inds, input_shape[1])
    rnd_mask = tf.reduce_sum(rnd_mask, 1)

    # Change values in original mask to be visible
    mask_in_rnd_sme = tf.minimum(same_mask + rnd_mask, 1.)
    mask_in_rnd_sme  = tf.minimum(mask_out, 1.-mask_in_rnd_sme)

    # We take only what we need (FOR FUTURE WORKS ON MULTIBAND CHANGE THIS)
    magnitudes = tf.slice(batch['input'], [0,0,1], [-1,-1, 1])

    # Choose random values and replace masked ones
    rnd_elem = tf.random.normal(tf.shape(rnd_mask), 0, tf.math.reduce_std(magnitudes, 1))
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
    batch['input_modified'] = magnitudes_copy
    batch['mask_in']  = tf.expand_dims(mask_in_rnd_sme, 2)
    batch['mask_out'] = tf.expand_dims(mask_out, 2)

    return batch

def mask_dataset(dataset,
                 msk_frac=.5,
                 rnd_frac=.2,
                 same_frac=.2):
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

    dataset = dataset.map(lambda x: mask_batch(x,
                                               msk_frac=msk_frac,
                                               rnd_frac=rnd_frac,
                                               same_frac=same_frac),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset
