import tensorflow as tf
import pandas as pd
import argparse
import logging
import json
import time
import os

from core.data import classification_records
from core.astromer import get_ASTROMER
from core.astromer_clf import get_CLASSIFIER, train

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def get_folder_name(path, prefix='model'):
    folders = [f for f in os.listdir(path)]
    if not folders:
        path = os.path.join(path, '{}_0'.format(prefix))
    else:
        n = sorted([int(f.split('_')[-1]) for f in folders])[-1]
        path = os.path.join(path, '{}_{}'.format(prefix, n+1))
    return path

def run(opt):

    # Check for pretrained weigths
    if os.path.isfile(os.path.join(opt.astromer, 'checkpoint')):
        print('[INFO] Using {}'.format(opt.astromer))
        conf_file = os.path.join(opt.astromer, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)
        # Loading hyperparameters of the pretrained model
        astromer = get_ASTROMER(num_layers=conf['layers'],
                                d_model=conf['head_dim'],
                                num_heads=conf['heads'],
                                dff=conf['dff'],
                                base=conf['base'],
                                dropout=conf['dropout'],
                                maxlen=conf['max_obs'])
        # Loading pretrained weights
        weights_path = '{}/weights'.format(opt.astromer)
        astromer.load_weights(weights_path)
        # Defining a new ()--p)roject folder
        opt.p = os.path.join(opt.astromer, 'classification')
        os.makedirs(opt.p, exist_ok=True)
        # Make sure we don't overwrite a previous training
        opt.p = get_folder_name(opt.p, prefix='rnn')
        # Creating (--p)royect directory
        os.makedirs(opt.p, exist_ok=True)
        # Init model
        num_cls = pd.read_csv(os.path.join(opt.data, 'objects.csv'))['label'].shape[0]
        clf = get_CLASSIFIER(astromer, opt.units, opt.dropout, num_cls)

        # tf.keras.utils.plot_model(
        #     clf, to_file='model.png', show_shapes=True
        # )

        # Loading data
        train_batches = classification_records(os.path.join(opt.data, 'train'),
                                               opt.batch_size,
                                               max_obs=opt.max_obs,
                                               take=opt.take)
        valid_batches = classification_records(os.path.join(opt.data, 'val'),
                                               opt.batch_size,
                                               max_obs=opt.max_obs,
                                               take=opt.take)

        train(clf, train_batches, valid_batches,
              epochs=opt.epochs, patience=opt.patience,
              verbose=0)

        # conf_file = os.path.join(opt.p, 'conf.json')
        # with open(conf_file, 'w') as json_file:
        #     json.dump(vars(opt), json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=50, type=int,
                    help='Max number of observations')
    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=2000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=200, type=int,
                        help='batch size')
    parser.add_argument('--take', default=1, type=int,
                        help='number of times to repeat the training and validation dataset')
    parser.add_argument('--astromer',  type=str,
                        help='ASTROMER pretrained weigths')

    # RNN HIPERPARAMETERS
    parser.add_argument('--units', default=16, type=int,
                        help='Number of self-attention heads')
    parser.add_argument('--dropout', default=0.2 , type=float,
                        help='dropout_rate for the encoder')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')

    opt = parser.parse_args()

    run(opt)
