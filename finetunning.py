import tensorflow as tf
import argparse
import logging
import json
import time
import os

from core.data  import load_records
from core.transformer import ASTROMER
from core.scheduler import CustomSchedule
from core.callbacks import get_callbacks
from core.losses import CustomMSE, ASTROMERLoss, CustomBCE
from core.metrics import CustomACC


def train(opt):

    conf_file = os.path.join(opt.exp, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    train_batches = load_records(os.path.join(conf['data'], 'test'),
                                 conf['batch_size'],
                                 magn_normed=conf['magn_normed'],
                                 time_normed=conf['time_normed'],
                                 shifted=conf['time_shifted'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TRAINING PAREMETERS
    parser.add_argument('--exp', default='./experiments/last_808', type=str,
                        help='Experiment folder')
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')

    parser.add_argument('--batch-size', default=512, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='Number of epochs')

    opt = parser.parse_args()

    train(opt)
