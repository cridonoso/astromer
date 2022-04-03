import tensorflow as tf
import pandas as pd
import subprocess
import argparse
import json
import os, sys

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from core.data import load_dataset, inference_pipeline

from presentation.experiments.clf.classifiers import build_lstm, \
                                                     build_lstm_att, \
                                                     build_mlp_att
from core.astromer import ASTROMER


def finetuning(data, astroweights, gpu):
    command1 = 'python -m presentation.experiments.clf.finetuning \
               {} \
               {} \
               {}'.format(gpu, data, astroweights)
    subprocess.call(command1, shell=True)

def classification(data, gpu):
    command1 = 'python -m presentation.experiments.clf.classify \
                        {} \
                        {}'.format(gpu, data)
    subprocess.call(command1, shell=True)

def run(opt):

    finetuning(opt.data, opt.w, opt.gpu)

    classification(opt.data, opt.gpu)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--data', default='alcock', type=str,
                        help='Dataset name')
    parser.add_argument('--w', default='./weights/macho', type=str,
                        help='ASTROMER pretrained weights')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU to use')
    opt = parser.parse_args()
    run(opt)
