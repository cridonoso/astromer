import tensorflow as tf
import pandas as pd
import subprocess
import argparse
import json
import os, sys


def finetuning(data, astroweights, gpu, case, astromer_size):
    command1 = 'python -m presentation.experiments.clf.finetuning \
               {} \
               {} \
               {} \
               {} \
               {}'.format(gpu, data, astroweights, case, astromer_size)
    subprocess.call(command1, shell=True)

def classification(data, gpu, case, astromer_size):
    command1 = 'python -m presentation.experiments.clf.classify \
                        {} \
                        {} \
                        {} \
                        {}'.format(gpu, data, case, astromer_size)
    subprocess.call(command1, shell=True)

def run(opt):

    finetuning(opt.data, opt.w, opt.gpu, opt.case, opt.size)

#     classification(opt.data, opt.gpu, opt.case, opt.size)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--data', default='alcock', type=str,
                        help='Dataset name')
    parser.add_argument('--case', default='a', type=str,
                        help='experiment scenario')
    parser.add_argument('--w', default='./weights/macho', type=str,
                        help='ASTROMER pretrained weights')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU to use')
    parser.add_argument('--size', default='256', type=str,
                        help='ASTROMER SIZE')
    opt = parser.parse_args()
    run(opt)
