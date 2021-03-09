import tensorflow as tf 
from tensorflow.keras.losses import BinaryCrossentropy

class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, name="RMSE"):
        super(CustomMSE, self).__init__(name=name)

    def call(self, y_true, y_pred, sample_weight=None):
        '''
        y_true = (batch_size, steps, 2)
        y_pred = (batch_size, steps, 2) being the last dimension the mask
        '''
        rec_pred = tf.slice(y_pred, [0,2,0], [-1, -1, 1])
        rec_mask = tf.slice(y_pred, [0,2,1], [-1, -1, 1])
        rec_true = tf.slice(y_true, [0,1,1], [-1, -1, -1])
        
        mse = tf.square(rec_true - rec_pred)
        # mse = tf.math.sqrt(mse)
        masked_mse = tf.multiply(mse, rec_mask)
        masked_mse = tf.reduce_sum(masked_mse, 1)

        return tf.reduce_mean(masked_mse)
        
class CustomBCE(tf.keras.losses.Loss):
    def __init__(self, name="BCE"):
        super(CustomBCE, self).__init__(name=name)

    def call(self, y_true, y_pred, sample_weight=None):
        '''
        y_true = (batch_size, steps, 2)
        y_pred = (batch_size, steps, 2) being the last dimension the mask
        '''
        cls_pred = tf.slice(y_pred, [0,0,0], [-1, 2, 1])
        cls_true = tf.slice(y_true, [0,0,0], [-1, 1, 1])
        

        y_one = tf.one_hot(tf.cast(tf.squeeze(cls_true), dtype=tf.int32), 2)


        bce = tf.nn.softmax_cross_entropy_with_logits(y_one, 
                                                      tf.squeeze(cls_pred))
        return bce

class ASTROMERLoss(tf.keras.losses.Loss):
    def __init__(self, name="AstromerLOSS"):
        super(ASTROMERLoss, self).__init__(name=name)
        self.mse = CustomMSE()
        self.bce = CustomBCE()

    def call(self, y_true, y_pred, sample_weight=None):
        rmse = self.mse(y_true, y_pred)
        
        bce = self.bce(y_true, y_pred)
        total = tf.reduce_mean(rmse) + tf.reduce_mean(bce)
        return total
