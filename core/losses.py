import tensorflow as tf 
from tensorflow.keras.losses import BinaryCrossentropy

class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, name="MSE"):
        super(CustomMSE, self).__init__(name=name)

    def call(self, y_true, y_pred):
        '''
        y_true = (batch_size, steps, 2)
        y_pred = (batch_size, steps, 2) being the last dimension the mask
        '''
        rec_pred = tf.slice(y_pred, [0,1,0], [-1, -1, 1])
        rec_mask = tf.slice(y_pred, [0,1,1], [-1, -1, 1])
        rec_true = tf.slice(y_true, [0,1,1], [-1, -1, -1])
        
        mse = tf.square(rec_true - rec_pred)
        mse = tf.math.sqrt(mse)
        masked_mse = tf.multiply(mse, rec_mask)
        masked_mse = tf.reduce_sum(masked_mse, 1)

        return tf.reduce_mean(masked_mse)
        
class CustomBCE(tf.keras.losses.Loss):
    def __init__(self, name="BCE"):
        super(CustomBCE, self).__init__(name=name)
        self.lossobject = BinaryCrossentropy(from_logits=False, 
                                             reduction='none')

    def call(self, y_true, y_pred):
        '''
        y_true = (batch_size, steps, 2)
        y_pred = (batch_size, steps, 2) being the last dimension the mask
        '''
        cls_pred = tf.slice(y_pred, [0,0,0], [-1, 1, 1])
        cls_true = tf.slice(y_true, [0,0,0], [-1, 1, 1])
        bce = self.lossobject(cls_true, cls_pred)
        return bce

class ASTROMERLoss(tf.keras.losses.Loss):
    def __init__(self, name="AstromerLOSS"):
        super(ASTROMERLoss, self).__init__(name=name)
        self.mse = CustomMSE()
        self.bce = CustomBCE()

    def call(self, y_true, y_pred):
        rmse = self.mse(y_true, y_pred)
        
        bce = self.bce(y_true, y_pred)
        total = tf.reduce_sum(rmse) + tf.reduce_sum(bce)
        return total
