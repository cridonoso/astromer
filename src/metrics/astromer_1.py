import tensorflow as tf

from tensorflow.keras.metrics import Metric

class MaskedRSquare(Metric):

    def __init__(self, name='masked_rsquare', **kwargs):
        super(MaskedRSquare, self).__init__(name=name, **kwargs)
        self.r2 = self.add_weight(name='r2', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        SS_res = tf.math.square(y_true - y_pred)
        SS_res =  tf.reduce_sum(SS_res*sample_weight)

        valid_true = y_true*sample_weight
        valid_mean = tf.math.divide_no_nan(tf.reduce_sum(valid_true, axis=1),
                                           tf.reduce_sum(sample_weight, axis=1))
        
        SS_tot = tf.math.square(y_true - tf.expand_dims(valid_mean, 1))
        SS_tot = tf.reduce_sum(SS_tot*sample_weight)
        values =  1.-tf.math.divide_no_nan(SS_res, SS_tot)

        self.r2.assign(values)

    def result(self):
        return self.r2