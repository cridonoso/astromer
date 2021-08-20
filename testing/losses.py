import tensorflow as tf
import unittest
from core.losses import custom_rmse
from core.data import pretraining_records
from core.astromer import get_ASTROMER

class TestStringMethods(unittest.TestCase):

    def test_dimensions(self):
        path_record = './data/records/macho/train'
        dataset = pretraining_records(path_record,
                                      batch_size=16,
                                      max_obs=100,
                                      repeat=1,
                                      msk_frac=0.5,
                                      rnd_frac=0.2,
                                      same_frac=0.2)


        # model = get_ASTROMER()
        for batch in dataset:
            print(batch['obserr'])
            # x_pred = model(batch)

            break

if __name__ == '__main__':
    unittest.main()
