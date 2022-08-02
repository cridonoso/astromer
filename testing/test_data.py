import tensorflow as tf
import numpy as np
import pytest
import sys

sys.path.append('/home')

from core.data import load_records, load_numpy, pretraining_pipeline
from core.data import mask_dataset, to_windows, mask_batch

def test_load_records():
    rec_dir = './data/records/alcock/fold_0/alcock_20/test'
    dataset = load_records(rec_dir)

    keys = list(dataset.element_spec.keys())
    expected = ['lcid', 'length', 'label', 'input']

    num_elements = sum([1 for _ in dataset])

    assert keys==expected, 'Dataset does not contains all labels'

def test_load_numpy():

    samples = [np.random.normal(size=[200, 3]),
               np.random.normal(size=[200, 3]),
               np.random.normal(size=[200, 3])]

    dataset = load_numpy(samples)

    keys = list(dataset.element_spec.keys())
    expected = ['lcid', 'length', 'label', 'input']

    num_elements = sum([1 for _ in dataset])

    assert sorted(keys)==sorted(expected), 'Dataset does not contains all labels'


@pytest.fixture
def test_to_windows():
    samples = [
    np.vstack([np.arange(100),
               np.arange(100),
               np.arange(100)]).T,
    np.vstack([np.arange(8),
               np.arange(8),
               np.arange(8)]).T,
    np.vstack([np.arange(10),
               np.arange(10),
               np.arange(10)]).T
    ]

    dataset = load_numpy(samples)
    dataset = to_windows(dataset,
                         2,
                         window_size=10,
                         sampling=True)

    for step, x in enumerate(dataset):
        assert x['input'].shape[1] == 10, 'Wrong window length'
        assert x['mask'].shape[1] == 10, 'Wrong masking'
        if step == 0:
            assert sum(x['mask'][1]) < 10, 'Wrong padding'

    return dataset



def test_old_new():
    samples = [
    np.vstack([np.arange(100),
               np.arange(100),
               np.arange(100)]).T,
    np.vstack([np.arange(8),
               np.arange(8),
               np.arange(8)]).T,
    np.vstack([np.arange(10),
               np.arange(10),
               np.arange(10)]).T
    ]

    data_old = pretraining_pipeline(samples,
                                    batch_size=2,
                                    window_size=10,
                                    rnd_frac=0.,
                                    shuffle=False,
                                    per_sample_mask=True)
    data_new = pretraining_pipeline(samples,
                                    batch_size=2,
                                    window_size=10,
                                    rnd_frac=0.,
                                    shuffle=False,
                                    per_sample_mask=False)

    for (x_old, y_old), (x_new, y_new) in zip(data_old.take(1),
                                              data_new.take(1)):
        v0 = x_old['input']
        v1 = x_new['input']
        print(v0)
        print(v1)

def test_mask_dataset(test_to_windows):
    dataset = mask_dataset(test_to_windows)
    for step, x in enumerate(dataset):
        if step == 0:
            assert sum(x['mask_out'][1][-2:]) == 0., 'Error when masking a padded light curve'
            assert sum(x['mask_in'][1][-2:]) == 2., 'Error when masking a padded light curve'
            assert sum(x['mask_out'][0]) == 5, 'Error in mask_out length'
            assert sum(x['mask_in'][0]) < 5, 'mask_in should be shorter than mask_out'

def test_pretraining_pipeline():
    rec_dir = './data/records/alcock/fold_0/alcock_20/test'
    dataset = pretraining_pipeline(rec_dir,batch_size=2,
                                   per_sample_mask=True)
    dataset_1 = pretraining_pipeline(rec_dir,batch_size=2, repeat=2,
                                   per_sample_mask=False)

    num_elements   = sum([1 for _ in dataset])
    num_elements_2 = sum([1 for _ in dataset_1])

    assert num_elements*2 == num_elements_2, 'repeat not working'
