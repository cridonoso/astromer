import tensorflow as tf

@tf.function
def custom_rmse(y_true, y_pred, mask=None):
    inp_shp = tf.shape(y_true)
    
    residuals = tf.square(y_true - y_pred)
    residuals = tf.multiply(residuals, mask)
    residuals  = tf.reduce_sum(residuals, 1)
    mse_mean = tf.math.divide_no_nan(residuals, tf.reduce_sum(mask, 1))
    mse_mean = tf.reduce_mean(mse_mean)
    return tf.math.sqrt(mse_mean)
#     return mse_mean