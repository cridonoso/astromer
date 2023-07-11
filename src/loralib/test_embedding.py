import unittest
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.ops import embedding_ops
from tensorflow.keras import layers, initializers
from layers import *
import numpy as np

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.embedding = Embedding(num_embeddings=10, embedding_dim=5, r=2, lora_alpha=2, merge_weights=True)
        self.embedding.build((None,))
        self.embedding(tf.constant([1, 2, 3]))  # Call once to initialize weights

    def test_init(self):
        self.assertEqual(self.embedding.num_embeddings, 10)
        self.assertEqual(self.embedding.embedding_dim, 5)
        self.assertEqual(self.embedding.r, 2)
        self.assertEqual(self.embedding.lora_alpha, 2)
        self.assertEqual(self.embedding.merge_weights, True)

    def test_reset_parameters(self):
        self.embedding.reset_parameters()
        self.assertTrue(np.all(self.embedding.lora_A.numpy() == 0))
        self.assertFalse(np.all(self.embedding.lora_B.numpy() == 0))
        self.assertFalse(np.all(self.embedding.embeddings.numpy() == 0))

    def test_call(self):
        inputs = tf.constant([1, 2, 3])
        outputs = self.embedding(inputs)
        self.assertEqual(outputs.shape, (3, 5))

    def test_train(self):
        self.embedding.trainable = True
        inputs = tf.constant([1, 2, 3])
        outputs = self.embedding(inputs, training=True)
        self.assertEqual(outputs.shape, (3, 5))

if __name__ == "__main__":
    test_util.run_all_in_graph_and_eager_modes(unittest.main())
