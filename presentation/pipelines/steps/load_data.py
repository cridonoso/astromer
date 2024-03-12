import tensorflow as tf
import pandas as pd
import os

from src.data.zero import pretraining_pipeline
from src.data import get_loader

def build_loader(data_path, params, batch_size=5, clf_mode=False, debug=False):

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
        nsp_prob = params['nsp_prob']
        
    train_loader = get_loader(os.path.join(data_path, 'train'),
                              batch_size=batch_size,
                              window_size=params['window_size'],
                              probed_frac=probed,
                              random_frac=random,
                              same_frac=same,
                              nsp_prob=nsp_prob,
                              sampling=False,
                              shuffle=True,
                              repeat=1,
                              aversion=params['arch'],
                              num_cls=num_cls)

    valid_loader = get_loader(os.path.join(data_path, 'val'),
                              batch_size=batch_size,
                              window_size=params['window_size'],
                              probed_frac=probed,
                              random_frac=random,
                              same_frac=same,
                              nsp_prob=nsp_prob,
                              sampling=False,
                              shuffle=False,
                              repeat=1,
                              aversion=params['arch'],
                              num_cls=num_cls)

    test_loader = get_loader(os.path.join(data_path, 'test'),
                              batch_size=batch_size,
                              window_size=params['window_size'],
                              probed_frac=probed,
                              random_frac=random,
                              same_frac=same,
                              nsp_prob=nsp_prob,
                              sampling=False,
                              shuffle=False,
                              repeat=1,
                              aversion=params['arch'],
                              num_cls=num_cls)
    
    return {
        'train': train_loader.take(2) if debug else train_loader,
        'validation': valid_loader.take(2) if debug else valid_loader,
        'test': test_loader.take(2) if debug else test_loader,
        'n_classes': num_cls,
    }