import tensorflow as tf
import numpy as np
import pytest
import sys

sys.path.append('/home')

from core.data import load_records, standardize
from core.data import mask_dataset, to_windows, nsp_dataset

def test_mask_dataset():
    rec_dir = './data/records/alcock/fold_0/alcock_20/test'
    dataset = load_records(rec_dir)
    dataset = to_windows(dataset, window_size=10, sampling=True)
    dataset = dataset.map(standardize)
    shapes = {'input' :[None, 3],
              'lcid'  :(),
              'length':(),
              'mask'  :[None, ],
              'label' :()}
    dataset = mask_dataset(dataset,
                           msk_frac=.5,
                           rnd_frac=.2,
                           same_frac=.2,
                           per_sample_mask=True,
                           window_size=10)
    shapes['input_modified'] = [None, None]
    shapes['mask_in'] = [None, None]
    shapes['mask_out'] = [None, None]

    dataset = dataset.padded_batch(4, padded_shapes=shapes)

    dataset = nsp_dataset(dataset)
