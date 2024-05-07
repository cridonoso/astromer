import tensorflow as tf
import pandas as pd
import os

from src.data.zero import pretraining_pipeline
from src.data import get_loader

def build_loader(data_path, params, batch_size=5, 
                 clf_mode=False, debug=False, 
                 normalize='zero-mean', 
                 sampling=False,
                 repeat=1,
                 old_version=False,
                 return_test=False):
    
    if clf_mode:
        print('Classification Mode')
        num_cls = pd.read_csv(os.path.join(data_path, 'objects.csv')).shape[0]
        probed = 1.
        random = 0.
        nsp_prob = 0. 
        same = 0.
    else:
        num_cls = None
        probed = params['probed']
        random = params['rs']
        try:
            same = params['same']
        except:
            same  = None
        try:
            nsp_prob = params['nsp_prob']
        except:
            nsp_prob = 0.
    

    if not 'norm' in list(params.keys()):
        norm = 'zero-mean'
    else:
        norm = params['norm']

    if old_version:        
        train_loader = pretraining_pipeline(os.path.join(data_path, 'train'),
                                 batch_size=batch_size,
                                 window_size=params['window_size'],
                                 msk_frac=probed,
                                 rnd_frac=random,
                                 same_frac=same,
                                 sampling=sampling,
                                 shuffle=True,
                                 repeat=repeat,
                                 num_cls=num_cls,
                                 normalize='zero-mean', # 'minmax'
                                 cache=False,
                                 return_ids=False,
                                 return_lengths=False,
                                 key_format=params['arch'])
        valid_loader = pretraining_pipeline(os.path.join(data_path, 'validation'),
                                 batch_size=batch_size,
                                 window_size=params['window_size'],
                                 msk_frac=probed,
                                 rnd_frac=random,
                                 same_frac=same,
                                 sampling=sampling,
                                 shuffle=False,
                                 repeat=1,
                                 num_cls=num_cls,
                                 normalize='zero-mean', # 'minmax'
                                 cache=False,
                                 return_ids=False,
                                 return_lengths=False,
                                 key_format=params['arch'])
        if return_test:
            test_loader = get_loader(os.path.join(data_path, 'test'),
                                      batch_size=batch_size,
                                      window_size=params['window_size'],
                                      probed_frac=probed,
                                      random_frac=random,
                                      same_frac=same,
                                      nsp_prob=nsp_prob,
                                      sampling=sampling,
                                      shuffle=False,
                                      normalize=norm,
                                      repeat=1,
                                      aversion=params['arch'],
                                      num_cls=num_cls)
    else:
        train_loader = get_loader(os.path.join(data_path, 'train'),
                                  batch_size=batch_size,
                                  window_size=params['window_size'],
                                  probed_frac=probed,
                                  random_frac=random,
                                  same_frac=same,
                                  nsp_prob=nsp_prob,
                                  sampling=sampling,
                                  shuffle=True,
                                  normalize=norm,
                                  repeat=repeat,
                                  aversion=params['arch'],
                                  num_cls=num_cls)
        
        valid_loader = get_loader(os.path.join(data_path, 'validation'),
                                  batch_size=batch_size,
                                  window_size=params['window_size'],
                                  probed_frac=probed,
                                  random_frac=random,
                                  same_frac=same,
                                  nsp_prob=nsp_prob,
                                  sampling=sampling,
                                  shuffle=False,
                                  normalize=norm,
                                  repeat=1,
                                  aversion=params['arch'],
                                  num_cls=num_cls)
        if return_test:
            test_loader = get_loader(os.path.join(data_path, 'test'),
                                      batch_size=batch_size,
                                      window_size=params['window_size'],
                                      probed_frac=probed,
                                      random_frac=random,
                                      same_frac=same,
                                      nsp_prob=nsp_prob,
                                      sampling=sampling,
                                      shuffle=False,
                                      normalize=norm,
                                      repeat=1,
                                      aversion=params['arch'],
                                      num_cls=num_cls)
        
    if return_test:
        return {
            'train': train_loader.take(2) if debug else train_loader,
            'validation': valid_loader.take(2) if debug else valid_loader,
            'test': test_loader.take(2) if debug else test_loader,
            'n_classes': num_cls,
        }
    else:
        return {
            'train': train_loader.take(2) if debug else train_loader,
            'validation': valid_loader.take(2) if debug else valid_loader,
            'n_classes': num_cls,
        }