import unittest
import tensorflow as tf

from core.data import sample_lc, mask_sample, load_dataset, pretraining_pipeline
from core.preprocess.masking import get_masked, set_random
from core.astromer import ASTROMER
from core.training.losses import custom_rmse
from core.training.metrics import custom_r2

class TestStringMethods(unittest.TestCase):

    def test_model_save(self):
        data = './data/records/testing/fold_0/testing/train'
        dataset = load_dataset(data)
        dataset = pretraining_pipeline(dataset,
                                       batch_size=256,
                                       max_obs=200,
                                       msk_frac=0.5,
                                       rnd_frac=0.2,
                                       same_frac=0.2)


        model = ASTROMER()
        model.load_weights('./testing/weights.h5')
        model.compile(optimizer='adam',
                      loss_rec=custom_rmse,
                      metric_rec=custom_r2)
        model.evaluate(dataset)






if __name__ == '__main__':
    unittest.main()
