import tensorflow as tf
import pandas as pd
import argparse
import logging
import json
import time
import os

from core.classifier import get_lstm_attention, get_lstm_no_attention, get_fc_attention, train
from core.data  import clf_records
from core.utils import get_folder_name
from time import gmtime, strftime

# os.system('clear')
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def run(opt):
    # Loading data
    print(opt.p)
    train_batches = clf_records(os.path.join(opt.data, 'train'),
                                opt.batch_size,
                                max_obs=opt.max_obs,
                                take=opt.take)
    valid_batches = clf_records(os.path.join(opt.data, 'val'),
                                opt.batch_size,
                                max_obs=opt.max_obs,
                                take=opt.take)

    num_classes = pd.read_csv(os.path.join(opt.data, 'objects.csv')).shape[0]


    if opt.mode == 0:
        clf = get_lstm_attention(opt.units,
                                 num_classes,
                                 opt.w,
                                 dropout=opt.dropout)
    if opt.mode == 1:
        clf = get_fc_attention(opt.units,
                               num_classes,
                               opt.w)
        print(clf.summary())
    if opt.mode == 2:
        clf = get_lstm_no_attention(opt.units,
                                    num_classes,
                                    maxlen=opt.max_obs,
                                    dropout=opt.dropout)

    # Creating (--p)royect directory
    os.makedirs(opt.p, exist_ok=True)
    
#     # Make sure we don't overwrite a previous training
#     opt.p = get_folder_name(opt.p, prefix='')

    # Save Hyperparameters
    conf_file = os.path.join(opt.p, 'conf.json')
    varsdic = vars(opt)
    varsdic['num_classes'] = int(num_classes)
    varsdic['exp_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    with open(conf_file, 'w') as json_file:
        json.dump(varsdic, json_file, indent=4)


    # Training ASTROMER
    train(clf, train_batches, valid_batches,
          patience=opt.patience,
          exp_path=opt.p,
          epochs=opt.epochs,
          lr=opt.lr,
          verbose=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')
    # ASTROMER
    parser.add_argument('--w', default="./runs/lr_1e-5", type=str,
                        help='ASTROMER pretrained weights')

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=200, type=int,
                        help='batch size')
    parser.add_argument('--take', default=100, type=int,
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
