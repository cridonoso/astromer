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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

@tf.function
def step(model, batch):
    encoder = model.get_layer('encoder')
    att = encoder(batch)
    return att

def run(opt):
    # Loading data
    batches = clf_records(opt.data,
                          opt.batch_size,
                          max_obs=opt.max_obs,
                          take=-1)

    
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

    os.makedirs('/'.join(opt.p.split('/')[:-1]), exist_ok=True)
    
    attention_vectors = []
    labels_vectors = []
    ids_vectors = []
    mask_vectors = []
    inp_vectors = []
    time_vectors = []
    inputs = []
    for batch in batches:
        att = step(model, batch)
        attention_vectors.append(att)
        labels_vectors.append(batch['label'])
        inp_vectors.append(batch['input'])
        time_vectors.append(batch['times'])
        mask_vectors.append(batch['mask_in'])
        ids_vectors.append(batch['lcid'])

    attention_vectors = tf.concat(attention_vectors, 0)
    inp_vectors       = tf.concat(inp_vectors, 0)
    time_vectors      = tf.concat(time_vectors, 0)
    labels_vectors    = tf.concat(labels_vectors, 0)
    mask_vectors      = tf.concat(mask_vectors, 0)
    ids_vectors       = tf.concat(ids_vectors, 0)
    
    
    hf = h5py.File(opt.p, 'w')
    hf.create_dataset('att', data=attention_vectors)
    hf.create_dataset('x', data=inp_vectors)
    hf.create_dataset('t', data=time_vectors)
    hf.create_dataset('y', data=labels_vectors)
    hf.create_dataset('id', data=ids_vectors.numpy().astype('S'))
    hf.create_dataset('m', data=mask_vectors)

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
