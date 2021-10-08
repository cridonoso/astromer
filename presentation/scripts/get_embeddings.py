import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import json
import os

from core.pretrained import ASTROMER_v1
from core.utils import get_folder_name


logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def run(opt):

    model = ASTROMER_v1(project_dir=opt.w)
    model.encode_from_records(os.path.join(opt.data, 'train'),
                              opt.batch_size,
                              dest=os.path.join(opt.p, 'train'))

    model.encode_from_records(os.path.join(opt.data, 'test'),
                              opt.batch_size,
                              dest=os.path.join(opt.p, 'test'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--w', default="./runs/debug", type=str,
                        help='pretrained model directory')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='folder for saving embeddings')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')

    opt = parser.parse_args()
    run(opt)
