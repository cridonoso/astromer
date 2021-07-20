import tensorflow as tf
import unittest
from core.data import pretraining_records
from core.masking import get_masked, set_random


class TestStringMethods(unittest.TestCase):

    def test_load(self):
        path_record = './data/records/macho/train'
        dataset = pretraining_records(path_record,
                                      batch_size=16,
                                      max_obs=200,
                                      repeat=1,
                                      msk_frac=0.5,
                                      rnd_frac=0.2,
                                      same_frac=0.2)

        for batch in dataset:
            batch_keys = list(batch.keys())
            break

        expected = ['lcid', 'length', 'label',
                    'input', 'output', 'times',
                    'mask_out', 'mask_in']

        self.assertListEqual(expected, batch_keys)


    def test_masked(self):
        msg = "Masked frac doesn't match the initial hyperparameter"
        seq_magn = tf.random.normal([50, 1])
        mask_out = get_masked(seq_magn, 0.5)
        frac = tf.reduce_sum(mask_out)/50
        self.assertEqual(frac, 0.5, msg)

    def test_random(self):
        x = tf.random.normal([50, 1])
        mask_out = get_masked(x, 0.5)

        n_masked = tf.reduce_sum(mask_out)
        seq_magn, mask_in = set_random(x,
                                       mask_out,
                                       tf.random.shuffle(x),
                                       rnd_frac=0.2,
                                       name='set_random')

        msg = "Random frac doesn't match the initial hyperparameter"
        self.assertTrue(tf.reduce_sum(mask_in) < n_masked)
        indices = tf.where(mask_out!=mask_in).numpy()
        n_random = indices.shape[0]/n_masked
        self.assertEqual(n_random, 0.2, msg)

        msg = 'No random element(s) detected'
        elms_0 = x.numpy()[indices]
        elms_1 = seq_magn.numpy()[indices]
        self.assertTrue(not any(e0 == e1 for e0, e1 in zip(elms_0, elms_1)), msg)

        msg = 'Masked values for random positions should be removed'
        m0 = tf.reduce_sum(mask_out.numpy()[indices])
        m1 = tf.reduce_sum(mask_in.numpy()[indices])
        self.assertGreater(m0, m1, msg)

        n_masked = tf.reduce_sum(mask_in)
        seq_magn, mask_in_2 = set_random(seq_magn,
                                         mask_in,
                                         seq_magn,
                                         rnd_frac=0.2,
                                         name='set_same')

        msg = "Same frac doesn't match the initial hyperparameter"
        self.assertTrue(tf.reduce_sum(mask_in_2) < n_masked)
        indices = tf.where(mask_in_2!=mask_in).numpy()
        m0 = tf.reduce_sum(mask_in.numpy()[indices])
        m1 = tf.reduce_sum(mask_in_2.numpy()[indices])

        msg = 'Masked values for random positions should be removed'
        self.assertGreater(m0, m1, msg)

    def test_dimensions(self):
        path_record = './data/records/macho/train'
        dataset = pretraining_records(path_record,
                                      batch_size=16,
                                      max_obs=200,
                                      repeat=1,
                                      msk_frac=0.5,
                                      rnd_frac=0.2,
                                      same_frac=0.2)

        for batch in dataset:
            for key in ['input', 'times', 'mask_in', 'mask_out']:
                current = batch[key][0]

                self.assertEqual(current.shape[-1], 1, '{} tensor should be [steps, 1]'.format(key))
            break




if __name__ == '__main__':
    unittest.main()
