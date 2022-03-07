import unittest
import tensorflow as tf
from core.data import sample_lc, mask_sample, load_dataset, pretraining_pipeline
from core.preprocess.masking import get_masked, set_random

class TestStringMethods(unittest.TestCase):
    def test_sample_random_window(self):
        '''
        Sample random window
        '''
        for try_obs in [50, 200, 1000]:
            input_dict = {
            'input': tf.transpose(
                        tf.stack([tf.range(0, try_obs),
                                  tf.range(0, try_obs),
                                  tf.range(0, try_obs)], 0)),
            'label': 0,
            'lcid': '0'
            }

            sequence, _, _ = sample_lc(input_dict, max_obs=200)
            cond_1 = sequence.shape[0] <= try_obs
            self.assertTrue(cond_1, 'window shape does not match')

        last = -1
        for i in range(2):
            sequence, _, _ = sample_lc(input_dict, max_obs=200)
            cond_2 = last != sequence[0][0]
            self.assertTrue(cond_2, 'Two identical windows were found')
            last = sequence[0][0]

    def test_set_random_or_same(self):
        seq_time = tf.expand_dims(tf.range(0, 10, dtype=tf.float32), 1)
        original = tf.expand_dims(tf.range(0, 10, dtype=tf.float32), 1)
        mask_out = tf.ones(10, dtype=tf.float32)


        seq_magn, mask_in = set_random(original,
                                       mask_out,
                                       original, # set same
                                       0.2,
                                       name='set_same')

        masked_0 = tf.boolean_mask(seq_magn, 1.-mask_in)
        masked_1 = tf.boolean_mask(original, 1.-mask_in)

        for i, j in zip(masked_0, masked_1):
            self.assertEqual(i,j,'vectors do not match after set_same mask applied')


        seq_magn, mask_in = set_random(original,
                                       mask_out,
                                       tf.random.shuffle(original), # set same
                                       0.2,
                                       name='set_same')

        masked_0 = tf.boolean_mask(seq_magn, 1.-mask_in)
        masked_1 = tf.boolean_mask(original, 1.-mask_in)

        cond = True
        for i, j in zip(masked_0, masked_1):
            if i != j:
                cond=True
        self.assertTrue(cond,'vectors do not match after set_random mask applied')

    def test_apply_masking_technique(self):
        input_dict = {
        'input': tf.transpose(
                    tf.stack([tf.range(0, 50, dtype=tf.float32),
                              tf.range(0, 50, dtype=tf.float32),
                              tf.range(0, 50, dtype=tf.float32)], 0)),
        'label': 0,
        'lcid': '0'
        }

        input_dict = mask_sample(input_dict['input'],
                             input_dict['label'],
                             input_dict['lcid'],
                             0.5, 0.2, 0.2, 200)

        mask_out_count = tf.reduce_sum(input_dict['mask_out'])

        self.assertTrue(mask_out_count >= 20,
                        'not enough masked values')

if __name__ == '__main__':
    unittest.main()
