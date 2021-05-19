import tensorflow as tf

@tf.function
def custom_acc(y_true, y_pred):
    y_pred  = tf.nn.softmax(tf.squeeze(y_pred))
    y_true = tf.reshape(y_true, [-1, 1])
    y_onehot = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
    y_onehot = tf.squeeze(y_onehot)

    acc = tf.keras.metrics.categorical_accuracy(y_onehot, y_pred)

    return tf.reduce_mean(acc)
