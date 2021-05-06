import tensorflow as tf

def custom_acc(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1, 1])
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.argmax(y_pred, 2)

    res = tf.math.equal(y_true, y_pred)

    res = tf.cast(res, tf.float32)

    return tf.reduce_mean(res)
