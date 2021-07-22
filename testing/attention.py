import tensorflow as tf
import unittest

from core.masking import reshape_mask

from core.attention import MultiHeadAttention

class TestStringMethods(unittest.TestCase):

    def test_self_attention(self):
        model_dim = 32
        mask = tf.convert_to_tensor([[0,0,0, 1,1,1, 0,0,0,0]])[..., tf.newaxis]
        mask = tf.cast(mask, tf.float32)
        x = tf.random.normal([1, len(mask[0]), model_dim])
        mask_rshp = reshape_mask(mask)

        mha = MultiHeadAttention(model_dim, 4)
        att, w = mha(x, mask_rshp)

        pos_w_zeros = tf.where(w[0][0][0]==0.)
        pos_m_zeros = tf.where(mask_rshp[0][0]==1.)


        self.assertListEqual(list(pos_w_zeros.numpy()[:,0]),
                             list(pos_m_zeros.numpy()[:,1]))


if __name__ == '__main__':
    unittest.main()
