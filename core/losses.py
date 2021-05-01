import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

@tf.function
def custom_mse(y_true, y_pred, sample_weight=None, mask=None):
    mse = tf.square(y_true - y_pred)

    if sample_weight is not None:
        mse = tf.multiply(mse, sample_weight)

    if mask is not None:
        mse = tf.multiply(tf.squeeze(mse), mask)

    mse = tf.reduce_sum(mse, 1)
    return tf.reduce_mean(mse)

@tf.function
def custom_bce(y_true, y_pred, num_cls=2, sample_weight=None):
    y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
    y_one = tf.one_hot(y_true, num_cls)
    y_pred = tf.squeeze(y_pred)
    bce = tf.nn.softmax_cross_entropy_with_logits(y_one, y_pred)

    return tf.reduce_mean(bce)
