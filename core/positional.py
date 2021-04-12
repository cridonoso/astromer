import tensorflow as tf
import numpy as np


# def get_angles(pos, i, d_model):
#     angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
#     return pos * angle_rates

# def positional_encoding(position, d_model):
#     angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                             np.arange(d_model)[np.newaxis, :],
#                             d_model)

#     # apply sin to even indices in the array; 2i
#     angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

#     # apply cos to odd indices in the array; 2i+1
#     angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

#     pos_encoding = angle_rads[np.newaxis, ...]

#     return tf.cast(pos_encoding, dtype=tf.float32)

def get_angles(times, d_model, base=10000):
    dim_indices = tf.range(d_model, dtype=tf.float32)

    exponent = tf.divide(tf.multiply(2., tf.divide(dim_indices, 2.)), 
                         tf.cast(d_model, tf.float32))

    angle_rates = tf.pow(tf.cast(base, dtype=tf.float32), exponent)
    angle_rates = tf.math.reciprocal(angle_rates)
    angle_rates = times * angle_rates
    return angle_rates


def positional_encoding(batch, d_model, base=10000, mjd=False):
    times = tf.slice(batch, [0, 0, 0], [-1, -1, 1], name='times')
    if mjd:
        indices = times
    else:
        indices = tf.range(tf.shape(times)[1], dtype=tf.float32)
        indices = tf.expand_dims(indices, 0)
        indices = tf.tile(indices, [tf.shape(times)[0], 1])
        indices = tf.expand_dims(indices, 2)

    angle_rads = get_angles(indices,
                            d_model,
                            base)

    # SIN AND COS
    def fn(x):
        if x[1] % 2 == 0:
            return (tf.sin(x[0]), x[1])
        else:
            return (tf.cos(x[0]), x[1])
        
    x_transpose = tf.transpose(angle_rads, [2,1,0])
    indices = tf.range(0, tf.shape(x_transpose)[0])
    x_transpose = tf.map_fn(lambda x: fn(x),  (x_transpose, indices))[0]
    pos_encoding = tf.transpose(x_transpose, [2, 1, 0])
    return tf.cast(pos_encoding, dtype=tf.float32)
