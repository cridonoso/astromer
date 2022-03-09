import unittest
import os
import tensorflow as tf

from core.data import sample_lc, mask_sample, load_dataset, pretraining_pipeline
from core.preprocess.masking import get_masked, set_random
from core.astromer import ASTROMER
from core.training.losses import custom_rmse
from core.training.metrics import custom_r2

os.environ["CUDA_VISIBLE_DEVICES"]='1'
def custom_rmse_2(y_true, y_pred, inp, sample_weight=None, mask=None):
    inp_shp = tf.shape(y_true)
    residuals = tf.square(y_true - y_pred)

    if sample_weight is not None:
        residuals = tf.multiply(residuals, sample_weight)

    if mask is not None:
        residuals = tf.multiply(residuals, mask)

    residuals  = tf.reduce_sum(residuals, 1)
    mse_mean = tf.divide(residuals,
                         tf.reduce_sum(mask, 1))
    
    for m, x, oid in zip(mask, y_true, inp):
        if tf.reduce_sum(m) == 0:
            print(m)
            print(x)
            print(oid)
            break
    mse_mean = tf.reduce_mean(mse_mean)
    
    return tf.math.sqrt(mse_mean)

class TestStringMethods(unittest.TestCase):

    def test_model_save(self):
        data = './data/records/big_atlas/fold_0/big_atlas/'
        train_ds = load_dataset(os.path.join(data, 'train'))
        train_ds = pretraining_pipeline(train_ds,
                                        batch_size=2048,
                                        max_obs=200,
                                        msk_frac=0.5,
                                        rnd_frac=0.2,
                                        same_frac=0.2)
        
        val_ds = load_dataset(os.path.join(data, 'val'))
        val_ds = pretraining_pipeline(val_ds,
                                        batch_size=2048,
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


        for x, (y, m), i in train_ds:
            y_pred = model(x)
            loss = custom_rmse_2(y, y_pred, inp=i, mask=m)
            if tf.math.is_nan(loss):
                break
if __name__ == '__main__':
    unittest.main()