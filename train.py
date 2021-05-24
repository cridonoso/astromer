import tensorflow as tf
import argparse
import logging
import json
import time
import os

from core.data import pretraining_records
from core.astromer import get_ASTROMER, get_FINETUNING, train

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
    # get_model
    astromer = get_ASTROMER(num_layers=opt.layers,
                            d_model=opt.head_dim,
                            num_heads=opt.heads,
                            dff=opt.dff,
                            base=opt.base,
                            dropout=opt.dropout,
                            maxlen=opt.max_obs)


    # Check for pretrained weigths
    if os.path.isfile(os.path.join(opt.p, 'checkpoint')):
        print('[INFO] Pretrained model detected! - Finetuning...')
        conf_file = os.path.join(opt.p, 'conf.json')
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
        weights_path = '{}/weights'.format(opt.p)
        astromer.load_weights(weights_path)
        # Defining a new ()--p)roject folder
        opt.p = os.path.join(opt.p, 'finetuning')
        os.makedirs(opt.p, exist_ok=True)
        # Make sure we don't overwrite a previous training
        opt.p = get_folder_name(opt.p, prefix='model')

    # Creating (--p)royect directory
    os.makedirs(opt.p, exist_ok=True)

    # Loading data
    train_batches = pretraining_records(os.path.join(opt.data, 'train'),
                                        opt.batch_size,
                                        max_obs=opt.max_obs)
    valid_batches = pretraining_records(os.path.join(opt.data, 'val'),
                                        opt.batch_size,
                                        max_obs=opt.max_obs)

    # Training ASTROMER
    train(astromer, train_batches, valid_batches,
          patience=opt.patience,
          exp_path=opt.p,
          epochs=opt.epochs,
          verbose=0)

    conf_file = os.path.join(opt.p, 'conf.json')
    with open(conf_file, 'w') as json_file:
        json.dump(vars(opt), json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=50, type=int,
                    help='Max number of observations')
    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./experiments/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=2000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=200, type=int,
                        help='batch size')
    parser.add_argument('--finetuning',default=False, action='store_true',
                        help='Finetune a pretrained model')
    parser.add_argument('--repeat', default=1, type=int,
                        help='number of times to repeat the training and validation dataset')
    # ASTROMER HIPERPARAMETERS
    parser.add_argument('--layers', default=1, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--heads', default=2, type=int,
                        help='Number of self-attention heads')
    parser.add_argument('--head-dim', default=812, type=int,
                        help='Head-attention Dimensionality ')
    parser.add_argument('--dff', default=256, type=int,
                        help='Dimensionality of the middle  dense layer at the end of the encoder')
    parser.add_argument('--dropout', default=0.1 , type=float,
                        help='dropout_rate for the encoder')
    parser.add_argument('--base', default=1000, type=int,
                        help='base of embedding')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')

    opt = parser.parse_args()
    opt.head_dim = (opt.max_obs + 3)*opt.heads
    run(opt)
