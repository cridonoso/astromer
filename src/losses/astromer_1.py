import tensorflow as tf 

from tensorflow.keras.losses import Loss

class MaskedMeanSquaredError(Loss):
    def __init__(self, *args, **kwargs):
        super(MaskedMeanSquaredError, self).__init__(*args, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        inp_shp = tf.shape(y_true)
        residuals = tf.square(y_true - y_pred)
        residuals = tf.multiply(residuals, sample_weight)
        residuals  = tf.reduce_sum(residuals, 1)
        mse_mean = tf.math.divide_no_nan(residuals, tf.reduce_sum(sample_weight, 1))
        mse_mean = tf.reduce_mean(mse_mean)
        return mse_mean
