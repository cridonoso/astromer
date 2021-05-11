import tensorflow as tf

def create_look_ahead_mask(size):
    mask = tf.math.subtract(1.,
                tf.linalg.band_part(tf.ones((size, size)), -1, 0, name='LowerTriangular'),
                name='LookaHeadMask')
    return mask  # (seq_len, seq_len)

def create_padding_mask(tensor, lengths):
    ''' Create mask given a tensor and true length '''
    with tf.name_scope("PaddingMask") as scope:
        lengths_transposed = tf.expand_dims(lengths, 1, name='Lengths')
        range_row = tf.expand_dims(tf.range(0, tf.shape(tensor)[1], 1), 0, name='Indices')
        # Use the logical operations to create a mask
        mask = tf.greater(range_row, lengths_transposed)
        return tf.cast(mask, tf.bool, name='LengthMask')

def create_MASK_token(tensor, frac=0.15):
    """ Add [MASK] values to be predicted
    Args:
        tensor : tensor values (only needed to get batch size and length)
        frac (float, optional): percentage for masking [MASK]
    Returns:
        binary tensor: a time-distributed mask
    """
    with tf.name_scope("Add_MASK_tokens") as scope:
        inp_size = tf.shape(tensor)
        def fn(x):
            probs = tf.math.log([[1-frac, frac]])
            m = tf.random.categorical(probs, inp_size[1], dtype=tf.int32)
            return m
        r = tf.map_fn(fn, tf.range(inp_size[0]))
        mask = tf.squeeze(r)
        return tf.cast(mask, dtype=tf.bool)

def set_random(tensor, mask, n_masked, frac=0.10):
    """ Insert random element in a tensor.
    They correspond to the "frac" portion of the total [MASK] tokens.
    Args:
        tensor (tensor): input values
        mask (binary tensor): [MASK] values
        frac (float, optional): percentage of the total to change by random
    """
    with tf.name_scope("set_random") as scope:
        if n_masked is None:
            n_masked = tf.reduce_sum(tf.cast(mask, tf.int32), 1)

        def fn(x):
            single_x, single_m, size = x
            n_rand = tf.multiply(size, frac)
            n_rand = tf.cast(n_rand, dtype=tf.int32)

            randvalues = tf.random.shuffle(single_x, name='shuffle_values')
            rand_elements = tf.slice(randvalues, [0, 0], [n_rand, -1])

            indices = tf.random.shuffle(tf.where(single_m), name='ShuffleMaskedIndices')
            selected = tf.slice(indices, [0, 0], [n_rand, -1])

            shape = tf.cast(tf.shape(single_x), tf.int64)

            rand_elements = tf.scatter_nd(selected, rand_elements, shape=shape)
            partial_mask = tf.math.divide_no_nan(rand_elements, rand_elements)
            partial_mask = tf.math.logical_not(tf.cast(partial_mask, tf.bool))
            partial_mask = tf.cast(partial_mask, tf.float32)

            new = tf.multiply(single_x, partial_mask)
            single_x = new + rand_elements

            single_m = tf.expand_dims(single_m, 1)
            single_m = tf.cast(single_m, tf.float32)*tf.slice(partial_mask, [0,0], [-1, 1])
            single_m = tf.cast(single_m, tf.bool)

            return [single_x, single_m, size]

        values, mask, _ = tf.map_fn(fn, [tensor, mask, n_masked])

        return values, mask

def set_same(mask, n_masked=None, frac=0.10):
    """Remove frac% of the [MASK] values
    Args:
        mask (TYPE): Description
        frac (float, optional): Description
    """
    with tf.name_scope("set_same") as scope:
        if n_masked is None:
            n_masked = tf.reduce_sum(tf.cast(mask, tf.int32), 1)

        def fn(x):
            single_m , size = x
            n_same = tf.multiply(size, frac)
            n_same = tf.cast(n_same, dtype=tf.int32)
            single_m = tf.cast(single_m, tf.float32)
            indices = tf.random.shuffle(tf.where(single_m == 1),
                                        name='ShuffleMaskedIndices')
            selected = tf.slice(indices, [0, 0], [n_same, -1])
            shape = tf.cast(tf.shape(single_m), tf.int64)
            partial_mask = tf.scatter_nd(selected, tf.ones(tf.shape(selected)[0], 1), shape=shape)
            partial_mask = tf.math.logical_not(tf.cast(partial_mask, tf.bool))
            partial_mask = tf.cast(partial_mask, tf.float32)
            single_m = single_m*partial_mask
            return [tf.cast(single_m, tf.bool), size]

        mask, _ = tf.map_fn(fn, [mask, n_masked])
        return mask

def concat_mask(mask1, mask2, cls, sep=102., reshape=True):
    ''' Concatenate Masks to build a BERT-style input'''
    with tf.name_scope("ConcatMask") as scope:
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
