import tensorflow as tf
import unittest
from tensorflow.python.framework import test_util
from layers import *

class TestLoRALayer(unittest.TestCase):
    def setUp(self):
        self.layer = LoRALayer(r=5, lora_alpha=2, lora_dropout=0.1, merge_weights=True)

    def test_init(self):
        self.assertEqual(self.layer.r, 5)
        self.assertEqual(self.layer.lora_alpha, 2)
        self.assertEqual(self.layer.merge_weights, True)
        self.assertEqual(self.layer.merged, False)
        self.assertTrue(isinstance(self.layer.lora_dropout, tf.keras.layers.Dropout))
    
    def test_dropout(self):
        input_tensor = tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)
        output_tensor = self.layer.lora_dropout(input_tensor)
        self.assertEqual(output_tensor.shape, input_tensor.shape)

if __name__ == "__main__":
    test_util.run_all_in_graph_and_eager_modes(unittest.main())
