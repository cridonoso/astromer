import tensorflow as tf

def create_look_ahead_mask(size):
    mask = tf.math.subtract(1., 
                tf.linalg.band_part(tf.ones((size, size)), -1, 0, name='LowerTriangular'),
                name='LookaHeadMask')
    return mask  # (seq_len, seq_len)

def create_padding_mask(tensor, lengths):
    ''' Create mask given a tensor and true length '''
    lengths_transposed = tf.expand_dims(lengths, 1)
    rangex = tf.range(0, tf.shape(tensor)[1], 1)
    range_row = tf.expand_dims(rangex, 0)
    # Use the logical operations to create a mask
    mask = tf.greater(range_row, lengths_transposed)
    return tf.cast(mask, tf.float32)

def create_prediction_mask(tensor, frac=0.15):
    time_steps = tf.shape(tensor)[1] 
    indices = tf.map_fn(lambda x: tf.random.shuffle(tf.range(time_steps))[:int(tf.cast(time_steps, tf.float32)*frac)], 
                        tf.range(tf.shape(tensor)[0]))
    one_hot = tf.one_hot(indices, time_steps)
    mask    = tf.reduce_sum(one_hot, 1)
    return tf.cast(mask, tf.float32)

def create_mask(tensor, length=None):
    mask = create_prediction_mask(tensor)
    if length is not None:
        mask_1 = create_padding_mask(tensor, length)
        mask = tf.maximum(mask, mask_1)

    return mask

def concat_mask(mask1, mask2, cls):
    cls = tf.tile(cls, [1, 1])
    sep = tf.tile([[102.]], [tf.shape(mask1)[0],1])
    mask = tf.concat([cls, mask1, sep, mask2], 1)
    dim_mask = tf.shape(mask)[1]
    mask = tf.tile(mask, [1, dim_mask])
    mask = tf.reshape(mask, [tf.shape(mask)[0], dim_mask, dim_mask])
    mask = tf.expand_dims(mask, 1)
    return mask