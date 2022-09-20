import tensorflow as tf

@tf.function
def custom_bce(y_true, y_pred, sample_weight=None):
    num_classes = tf.shape(y_pred)[-1]
    y_one = tf.one_hot(y_true, num_classes)
    y_one = tf.cast(y_one, tf.float32)
    
    losses = tf.nn.softmax_cross_entropy_with_logits(y_one, y_pred)

    return tf.reduce_mean(losses)
