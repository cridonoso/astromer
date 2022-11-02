import tensorflow as tf
import numpy as np
import pytest
import sys

sys.path.append('/home')

from src.data import load_records, standardize
from src.data import mask_dataset, to_windows, nsp_dataset
from src.models import get_ASTROMER_nsp

def test_mask_dataset():
    rec_dir = './data/records/alcock/fold_0/alcock_20/test'
    dataset = load_records(rec_dir)
    dataset = to_windows(dataset, window_size=200, sampling=True)
    dataset = dataset.map(standardize)
    dataset = mask_dataset(dataset,
                           msk_frac=.5,
                           rnd_frac=.2,
                           same_frac=.2,
                           window_size=200)

    dataset = nsp_dataset(dataset, buffer_shuffle=5000)

    shapes = {'input' :[None, 3],
              'lcid'  :(),
              'length':(),
              'mask'  :[None, None],
              'label' :(),
              'nsp_label':(),
              'input_modified': [None, None],
              'mask_in': [None, None],
              'mask_out': [None, None]}
    shapes['nsp_label'] = ()
    shapes['mask'] = (None, None)
    shapes['original_input'] = (None, 3)


    dataset = dataset.padded_batch(512, padded_shapes=shapes)


def test_model():
    model = get_ASTROMER_nsp(d_model=256, maxlen=100)
    model.summary()
