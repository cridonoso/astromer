import unittest
import tensorflow as tf

from core.data import sample_lc, mask_sample, load_dataset, pretraining_pipeline
from core.preprocess.masking import get_masked, set_random
from core.astromer import ASTROMER
from core.training.losses import custom_rmse
from core.training.metrics import custom_r2

class TestStringMethods(unittest.TestCase):

    def test_model_forward(self):
        data = './data/records/testing/fold_0/testing/train'
        dataset = load_dataset(data)
        dataset = pretraining_pipeline(dataset,
                                       batch_size=256,
                                       max_obs=200,
                                       msk_frac=0.5,
                                       rnd_frac=0.2,
                                       same_frac=0.2)


        model = ASTROMER()
        for x, y in dataset:
            y_pred = model(x)
            self.assertEqual(y_pred.shape[1], 200, 'Output should be 200 length')
            break


    def test_model_fit(self):
        data = './data/records/testing/fold_0/testing/train'
        dataset = load_dataset(data)
        dataset = pretraining_pipeline(dataset,
                                       batch_size=256,
                                       max_obs=200,
                                       msk_frac=0.5,
                                       rnd_frac=0.2,
                                       same_frac=0.2)


        model = ASTROMER()
        model.build({'input': [None, 200, 1],
                     'mask_in': [None, 200, 1],
                     'times': [None, 200, 1]})
        model.compile(optimizer='adam',
                      loss_rec=custom_rmse,
                      metric_rec=custom_r2)

        history = model.fit(dataset, epochs=2, validation_data=dataset)

if __name__ == '__main__':
    unittest.main()
