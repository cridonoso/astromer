import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.nn import (sigmoid_cross_entropy_with_logits,
                           softmax_cross_entropy_with_logits)

@tf.function
def custom_rmse(y_true, y_pred, sample_weight=None, mask=None):
    inp_shp = tf.shape(y_true)
    mse = tf.square(y_true - y_pred)

    if sample_weight is not None:
        mse = tf.multiply(mse, sample_weight)

    if mask is not None:
        mse = tf.multiply(mse, mask)

    mse = tf.reduce_sum(mse, 1)
    return tf.math.sqrt(tf.reduce_mean(mse))

@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    num_classes = tf.shape(y_pred)[-1]
    if len(tf.shape(y_pred)) > 2:
        num_steps = tf.shape(y_pred)[1]
        y_one = tf.one_hot(y_true, num_classes)
        y_one = tf.expand_dims(y_one, 1)
        y_one =tf.tile(y_one, [1,num_steps,1])
    else:
        num_steps = tf.shape(y_pred)[-1]
        y_one = tf.one_hot(y_true, num_classes)

    losses = tf.nn.softmax_cross_entropy_with_logits(y_one, y_pred)

    if len(tf.shape(y_pred)) > 2:
        losses = tf.transpose(losses)
        losses = tf.reduce_sum(losses, 1)

    return tf.reduce_mean(losses)
