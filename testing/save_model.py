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
        model.build({'input': [None, 200, 1],
                     'mask_in': [None, 200, 1],
                     'times': [None, 200, 1]})
        model.compile(optimizer='adam',
                      loss_rec=custom_rmse,
                      metric_rec=custom_r2)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='./testing/weights.h5',
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        history = model.fit(dataset,
                            epochs=2,
                            validation_data=dataset,
                            callbacks=[model_checkpoint_callback])






if __name__ == '__main__':
    unittest.main()
