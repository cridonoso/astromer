import tensorflow as tf
import unittest
from core.losses import custom_mse
from core.data import pretraining_records

class TestStringMethods(unittest.TestCase):

    def test_load(self):

        x_true = tf.ones([1, 10, 1])
        x_pred = tf.ones([1, 10, 1])-0.5
        mask = tf.convert_to_tensor([[0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]])[..., tf.newaxis]

        # rmse_1 = custom_mse(x_true, x_pred)
        rmse_2 = custom_mse(x_true, x_pred, mask)

        self.assertEqual(rmse_2, 0.5)

if __name__ == '__main__':
    unittest.main()
