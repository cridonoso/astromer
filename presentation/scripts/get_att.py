import tensorflow as tf
import pandas as pd
import argparse
import logging
import json
import time
import h5py
import os

from core.classifier import get_lstm_attention, get_lstm_no_attention, get_fc_attention, train
from core.data  import clf_records
from core.utils import get_folder_name
from time import gmtime, strftime
from core.astromer import get_ASTROMER

os.system('clear')
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

@tf.function
def step(model, batch):
    att = model(batch)
    return att

def run(opt):
    # Loading data
    batches = clf_records(opt.data,
                          opt.batch_size,
                          max_obs=opt.max_obs,
                          take=opt.take)


    conf_file = os.path.join(opt.w, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    model = get_ASTROMER(num_layers=conf['layers'],
                         d_model   =conf['head_dim'],
                         num_heads =conf['heads'],
                         dff       =conf['dff'],
                         base      =conf['base'],
                         dropout   =conf['dropout'],
                         maxlen    =conf['max_obs'],
                         use_leak  =conf['use_leak'])

    weights_path = '{}/weights'.format(opt.w)
    model.load_weights(weights_path)
    encoder = model.get_layer('encoder')

    attention_vectors = []
    labels_vectors = []
    lens_vectors = []
    for batch in batches:
        att = step(encoder, batch)
        attention_vectors.append(att)
        labels_vectors.append(batch['label'])
        lens_vectors.append(batch['mask_in'])

    att_train = tf.concat(attention_vectors, 0)
    lab_train = tf.concat(labels_vectors, 0)
    len_train = tf.concat(lens_vectors, 0)

    hf = h5py.File(opt.p, 'w')
    hf.create_dataset('x', data=att_train)
    hf.create_dataset('y', data=lab_train)
    hf.create_dataset('l', data=len_train)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')
    # ASTROMER
    parser.add_argument('--w', default="./runs/huge", type=str,
                        help='ASTROMER pretrained weights')

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/alcock', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')

    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=200, type=int,
                        help='batch size')
    parser.add_argument('--take', default=-1, type=int,
                        help='Number of balanced batches for training. -1 do not balance')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')
    # RNN HIPERPARAMETERS
    parser.add_argument('--units', default=256, type=int,
                        help='number of units for the RNN')
    parser.add_argument('--dropout', default=0.5 , type=float,
                        help='dropout_rate for the encoder')

    parser.add_argument('--mode', default=0, type=int,
                        help='Classifier model: 0: LSTM + ATT - 1: MLP + ATT - 2 LSTM')



    opt = parser.parse_args()
    run(opt)
