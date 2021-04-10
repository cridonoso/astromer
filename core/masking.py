import tensorflow as tf

def create_look_ahead_mask(size):
    mask = tf.math.subtract(1.,
                tf.linalg.band_part(tf.ones((size, size)), -1, 0, name='LowerTriangular'),
                name='LookaHeadMask')
    return mask  # (seq_len, seq_len)

def create_padding_mask(tensor, lengths):
    ''' Create mask given a tensor and true length '''
    lengths_transposed = tf.expand_dims(lengths, 1, name='Lengths')
    range_row = tf.expand_dims(tf.range(0, tf.shape(tensor)[1], 1), 0, name='Indices')
    # Use the logical operations to create a mask
    mask = tf.greater(range_row, lengths_transposed)
    return tf.cast(mask, tf.float32, name='LengthMask')

def get_mask(tensor, frac=0.15):
    '''
    Creates a random mask given an observation fraction
    '''
    time_steps = tf.shape(tensor, name='NumSteps')[1]
    indices = tf.map_fn(lambda x: tf.random.shuffle(tf.range(time_steps))[:int(tf.cast(time_steps, tf.float32)*frac)],
                        tf.range(tf.shape(tensor)[0]), name='RandIndices')
    one_hot = tf.one_hot(indices, time_steps, name='OneHotVec')
    mask    = tf.reduce_sum(one_hot, 1, name='SumOverOneHot')
    return tf.cast(mask, tf.float32)


def random_same_replacement(tensor, tensor_mask, frac_random=0.1, frac_same=0.1):
    allmagn = tf.slice(tensor, [0, 0, 1], [-1, -1, 1])
    allmagn = tf.reshape(allmagn, [tf.shape(allmagn)[0]*tf.shape(allmagn)[1]],
               name='FlattenMagns')

    def change_random(x):
        inputs, mask = x
        times = tf.slice(inputs, [0, 0], [-1, 1], name='times')
        magn = tf.slice(inputs, [0, 1], [-1, 1], name='mangitudes')

        size = tf.reduce_sum(mask, name='NumMaskedValues')
        n_rand = tf.multiply(size, frac_random)
        n_rand = tf.cast(n_rand, dtype=tf.int32)

        n_same = tf.multiply(size, frac_same)
        n_same = tf.cast(n_same, dtype=tf.int32)

        if n_rand > 0 :
            randmang = tf.random.shuffle(allmagn, name='ShuffleMagns')
            rand_elements = tf.slice(randmang, [0], [n_rand])

            indices = tf.random.shuffle(tf.where(mask == 1), name='ShuffleMaskedIndices')
            selected = tf.slice(indices, [0, 0], [n_rand, -1])

            rand_elements = tf.scatter_nd(selected, rand_elements, [tf.shape(magn)[0]])
            partial_mask = tf.scatter_nd(selected, [1.], [tf.shape(magn)[0]])
            partial_mask = tf.math.logical_not(tf.cast(partial_mask, tf.bool))
            partial_mask = tf.cast(partial_mask, tf.float32)
            new = tf.multiply(magn, tf.expand_dims(partial_mask, 1))
            new = new + tf.expand_dims(rand_elements, 1)

            inputs = tf.concat([times, new], 1)
            mask = mask*partial_mask


        if n_same > 0:
            indices = tf.random.shuffle(tf.where(mask == 1), name='ShuffleMaskedIndices')
            selected = tf.slice(indices, [0, 0], [n_same, -1])
            partial_mask = tf.scatter_nd(selected, [1.], [tf.shape(magn)[0]])
            partial_mask = tf.math.logical_not(tf.cast(partial_mask, tf.bool))
            partial_mask = tf.cast(partial_mask, tf.float32)
            mask = mask*partial_mask

        return [inputs, mask]



    tensor, tensor_mask = tf.map_fn(change_random, [tensor, tensor_mask])


    return tensor, tensor_mask

def create_mask(tensor, length=None, frac=0.15, frac_random=0.1, frac_same=0.1):
    ''' Creates random and variable length masks '''
    mask = create_prediction_mask(tensor, frac=frac)
    tensor, mask_inp = random_same_replacement(tensor, mask, frac_random=frac_random, frac_same=frac_same)

    if length is not None:
        mask_1 = create_padding_mask(tensor, length)
        mask_inp = tf.maximum(mask_inp, mask_1, name='CombinedMask')

    return mask, mask_inp

def concat_mask(mask1, mask2, cls, sep=102., reshape=True):
    ''' Concatenate Masks to build a BERT-style input'''
    cls = tf.tile(tf.ones_like(cls), [1, 1])
    if reshape:
        sep_token = tf.tile([[tf.ones_like(sep)]], [tf.shape(mask1)[0],1])
        mask = tf.concat([cls, mask1, sep_token, mask2], 1)
        dim_mask = tf.shape(mask)[1]
        mask = tf.tile(mask, [1, dim_mask])
        mask = tf.reshape(mask, [tf.shape(mask)[0], dim_mask, dim_mask])
        mask = tf.expand_dims(mask, 1)
    else:
        sep_token = tf.tile([[tf.zeros_like(sep)]], [tf.shape(mask1)[0],1])
        mask = tf.concat([cls, mask1, sep_token, mask2], 1)

    return mask
