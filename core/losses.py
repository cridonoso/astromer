import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

@tf.function
def custom_mse(y_true, y_pred, sample_weight=None, mask=None):
    mse = tf.square(tf.square(y_true - y_pred))

    if sample_weight is not None:
        mse = tf.multiply(mse, sample_weight)

    if mask is not None:
        d = tf.boolean_mask(mse[0], mask[0])
        mse = tf.multiply(tf.squeeze(mse), mask)

    mse = tf.reduce_sum(mse, 1)
    return tf.reduce_mean(mse)

@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(tf.squeeze(y_true), dtype=tf.int32)
    y_one = tf.one_hot(y_true, tf.shape(y_pred)[-1])

    bce = tf.nn.softmax_cross_entropy_with_logits(y_one, y_pred)

    return tf.reduce_mean(bce)
