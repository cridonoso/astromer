import tensorflow as tf
import numpy as np


def get_angles(times, d_model, base=10000):
    with tf.name_scope("Get_Angles") as scope:
        dim_indices = tf.range(d_model, dtype=tf.float32)

        exponent = tf.divide(tf.multiply(2., tf.divide(dim_indices, 2.)),
                             tf.cast(d_model, tf.float32))

        angle_rates = tf.pow(tf.cast(base, dtype=tf.float32), exponent)
        angle_rates = tf.math.reciprocal(angle_rates)
        angle_rates = times * angle_rates
        return angle_rates


def positional_encoding(times, d_model, base=10000, mjd=False):
    with tf.name_scope("PosEncoding") as scope:
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
