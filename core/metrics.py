import tensorflow as tf


@tf.function
def custom_acc(y_true, y_pred):
    if len(tf.shape(y_pred)) > 2:
        y_pred  = tf.nn.softmax(y_pred)[:,-1,:]
    else:
        y_pred  = tf.nn.softmax(y_pred)

    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.argmax(y_pred, 1, output_type=tf.int32)
    y_pred = tf.expand_dims(y_pred, 1)

    correct = tf.math.equal(y_true, y_pred)
    correct = tf.cast(correct, tf.float32)

    return tf.reduce_mean(correct)

@tf.function
def custom_rmse(y_true, y_pred):
    mask = tf.math.divide_no_nan(y_true, y_true)
    y_true = tf.cast(y_true, y_pred.dtype)
    num = tf.math.square(y_pred - y_true)
    num = tf.multiply(num, mask)
    num = tf.reduce_sum(num, axis=1)
    tot = tf.reduce_sum(mask, axis=1)
    return tf.math.sqrt(tf.math.divide_no_nan(num, tot))

@tf.function
def custom_rsquare(y_true, y_pred):
    mask = tf.math.divide_no_nan(y_true, y_true)
    unexplained_error = tf.square(y_true - y_pred)
    unexplained_error = tf.multiply(unexplained_error, mask)
    unexplained_error = tf.reduce_sum(unexplained_error)

    total_error = tf.square(y_true - tf.reduce_mean(y_true))
    total_error = tf.multiply(total_error, mask)
    total_error = tf.reduce_sum(total_error)

    R_squared = 1 - tf.divide(unexplained_error, total_error)

    return R_squared
