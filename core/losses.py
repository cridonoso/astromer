import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.nn import (sigmoid_cross_entropy_with_logits,
                           softmax_cross_entropy_with_logits)

@tf.function
def custom_mse(y_true, y_pred, septkn=-98, sample_weight=None, mask=None):
    inp_shp = tf.shape(y_true)
    sep_tkn = tf.cast([[[septkn]]], dtype=tf.float32)
    sep_tkn = tf.tile(sep_tkn, [inp_shp[0], 1, inp_shp[-1]])
    mask_tkn = tf.cast([[[1.]]], dtype=tf.float32)
    mask_tkn = tf.tile(mask_tkn, [inp_shp[0], 1, 1])

    yt1, yt2 = tf.split(y_true, 2, 1)
    y_true = tf.concat([yt1, sep_tkn, yt2, sep_tkn], 1)

    mse = tf.square(y_true - y_pred)

    if sample_weight is not None:
        mse = tf.multiply(mse, sample_weight)

    if mask is not None:
        m1, m2 = tf.split(mask, 2, 1)
        mask = tf.concat([m1, mask_tkn, m2, mask_tkn], 1)
        mse = tf.multiply(mse, mask)

    mse = tf.reduce_sum(mse, 1)
    return tf.math.sqrt(tf.reduce_mean(mse))

@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
    y_one = tf.one_hot(y_true, tf.shape(y_pred)[-1])

    bce = sigmoid_cross_entropy_with_logits(y_one, y_pred)
    return tf.reduce_mean(bce)*2
