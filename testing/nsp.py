import unittest
import tensorflow as tf
from core.data import load_dataset, pretraining_pipeline_nsp
from core.preprocess.masking import get_masked, set_random
from core.astromer import ASTROMER_NSP
from core.training.losses import custom_rmse
from core.training.metrics import custom_r2

import matplotlib.pyplot as plt

class TestStringMethods(unittest.TestCase):

    def test_load_pretraining(self):
        data = './data/records/testing/fold_0/testing/train'
        dataset = load_dataset(data, shuffle=True, repeat=1)

        dataset = pretraining_pipeline_nsp(dataset,
                                           batch_size=256,
                                           max_obs=200,
                                           msk_frac=0.5,
                                           rnd_frac=0.2,
                                           same_frac=0.2,
                                           nsp_proba=0.5)


        model = ASTROMER_NSP()
        model.build({'input': [None, 200, 1],
                     'mask_in': [None, 200, 1],
                     'times': [None, 200, 1]})
        model.compile(optimizer='adam')

        for x, (y, label, mask) in dataset:
            x_rec, y_pred = model(x)
            cce = model.binary_ce(label, y_pred)
            acc = model.accuracy(label, y_pred)
            print(acc)
            break

        #     plt.figure(figsize=(6,3))
        #     plt.plot(x['times'][0], '.-')
        #     plt.xlabel('TIME STEP')
        #     plt.ylabel('MJD')
        #     plt.title('stitch-fix')
        #     plt.show()
        #     break

if __name__ == '__main__':
    unittest.main()
