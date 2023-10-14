
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from attention import scaled_dot_product_attention, HeadAttentionMulti
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HeadAttentionMultiTest(tf.test.TestCase):

    def testScaledDotProductAttentionOutputCorrectness(self):
        batch_size = 1
        seq_len = 2
        depth = 1
        q = tf.ones([batch_size, seq_len, depth])
        k = tf.ones([batch_size, seq_len, depth]) + 1
        v = tf.ones([batch_size, seq_len, depth]) + 1
        output, attention_weights, _ = \
            scaled_dot_product_attention(q, k, v, mask=None, debug_mode=True)

        expected_output = np.array([[[2.], [2.]]]) # shape: (1, 2, 1)
        expected_attention_weight = np.array([[[0.5, 0.5], [0.5, 0.5]]])


        self.assertAllEqual(expected_output, output)
        self.assertAllEqual(expected_attention_weight, attention_weights)

    def testScaledDotProductAttentionTensorShapes(self):
        batch_size = 1
        seq_len = 2
        depth = 1
        q = tf.ones([batch_size, seq_len, depth])
        k = tf.ones([batch_size, seq_len, depth]) + 1
        v = tf.ones([batch_size, seq_len, depth]) + 1
        output, attention_weights, debug_dict = \
            scaled_dot_product_attention(q, k, v, mask=None, debug_mode=True)

        # Create dummy zeros arrays with expected dimensions
        self.assertShapeEqual(np.zeros((batch_size, seq_len, seq_len)), debug_dict["matmul_qk"]) #
        self.assertShapeEqual(np.zeros(()), debug_dict["dk"]) # 0 is a dummy value with shape ()
        self.assertShapeEqual(np.zeros((batch_size, seq_len, seq_len)), debug_dict["scaled_attention_logits"])
        self.assertShapeEqual(np.zeros((batch_size, seq_len, seq_len)), debug_dict["attention_weights"])
        self.assertShapeEqual(np.zeros((batch_size, seq_len, depth)), output)
        self.assertShapeEqual(np.zeros((batch_size, seq_len, seq_len)), attention_weights)

    

    def testMultiHeadAttentionOutputShapes(self):
        d_model = 4
        num_heads = 2
        head_dim = 2
        batch_size = 2
        seq_len = 4
        depth = d_model // num_heads

        debug_mode = True
        # def call(self, v, k, q, mask):
        print(head_dim, num_heads, debug_mode)
        multihead_attention = HeadAttentionMulti(head_dim, num_heads, debug_mode)

        q = tf.ones([batch_size, seq_len, depth])
        k = tf.ones([batch_size, seq_len, depth])
        v = tf.ones([batch_size, seq_len, depth])

        output, _, debug_dict = multihead_attention(v, k, q, training=True,mask=None)
        self.assertShapeEqual(np.zeros((batch_size, num_heads, seq_len, depth)), debug_dict["split_q"])
        self.assertShapeEqual(np.zeros((batch_size, num_heads, seq_len, depth)), debug_dict["split_k"])
        self.assertShapeEqual(np.zeros((batch_size, num_heads, seq_len, depth)), debug_dict["split_v"])
        self.assertShapeEqual(np.zeros((batch_size, seq_len, num_heads, depth)), debug_dict["scaled_attention"])
        self.assertShapeEqual(np.zeros((batch_size, seq_len, d_model)), debug_dict["concat_attention"])
        self.assertShapeEqual(np.zeros((batch_size, seq_len, d_model)), output)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


tf.test.main()
