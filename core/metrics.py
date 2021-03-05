import tensorflow as tf 

class CustomACC(tf.keras.metrics.Metric):
    def __init__(self, name="Accuracy", **kwargs):
        super(CustomACC, self).__init__(name=name, **kwargs)


        self.object = tf.keras.metrics.Accuracy(name='accuracy')

        self.acc_value = self.add_weight(name="value", 
                                         initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):     

        cls_pred = tf.slice(y_pred, [0,0,0], [-1, 1, -1])
        cls_true = tf.slice(y_true, [0,0,0], [-1, 1, 1])

        value = self.object(cls_true, cls_pred)
        self.acc_value.assign_add(value)


    def result(self):
        return self.acc_value


