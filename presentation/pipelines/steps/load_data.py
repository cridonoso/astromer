import tensorflow as tf
import pandas as pd
import os

from src.data import get_loader

def build_loader(data_path, params, batch_size=5, 
                 clf_mode=False, 
                 debug=False, 
                 normalize='zero-mean', 
                 sampling=False,
                 repeat=1,
                 return_test=False,
                 probed=None,
                 same=None,
                 random=None):
    
    norm = normalize if normalize is not None else params['norm']

    if clf_mode:
        print('Classification Mode')
        num_cls = pd.read_csv(os.path.join(data_path, 'objects.csv')).shape[0]
        probed  = 1.
        random  = 0.
        same    = 0.
    else:
        num_cls = None
        probed = probed if probed is not None else params['probed']
        random = random if random is not None else params['rs']
        same   = same   if same   is not None else params['same']

    
    val_path = os.path.join(data_path, 'validation')
    if not os.path.isdir(val_path):
        val_path = os.path.join(data_path, 'val')    
        print('[INFO] Changing path: ', val_path)


    train_loader = get_loader(os.path.join(data_path, 'train'),
                                batch_size=batch_size,
                                window_size=params['window_size'],
                                probed_frac=probed,
                                random_frac=random,
                                same_frac=same,
                                sampling=sampling,
                                shuffle=True,
                                normalize=norm,
                                repeat=repeat,
                                aversion=params['arch'],
                                num_cls=num_cls)
    
    valid_loader = get_loader(val_path,
                                batch_size=batch_size,
                                window_size=params['window_size'],
                                probed_frac=probed,
                                random_frac=random,
                                same_frac=same,
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
                                    sampling=False,
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