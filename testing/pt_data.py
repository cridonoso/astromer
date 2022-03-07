import unittest
import tensorflow as tf
from core.data import sample_lc, mask_sample, load_dataset, pretraining_pipeline
from core.preprocess.masking import get_masked, set_random

class TestStringMethods(unittest.TestCase):

    def test_load_pretraining(self):
        data = './data/records/testing/fold_0/testing/train'
        dataset = load_dataset(data)
        dataset = pretraining_pipeline(dataset,
                                       batch_size=256,
                                       max_obs=200,
                                       msk_frac=0.5,
                                       rnd_frac=0.2,
                                       same_frac=0.2)
        for x, y in dataset:
            print(x['mask_in'])
            break

if __name__ == '__main__':
    unittest.main()
