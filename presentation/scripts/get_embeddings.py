import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import logging
import h5py
import json
import os

from core.astromer import get_ASTROMER
from core.utils import get_folder_name
from core.data import load_records
from shutil import copyfile
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def run(opt):
    conf_file = os.path.join(opt.w, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)



    start_time = datetime.now()
    # Loading hyperparameters of the pretrained model
    astromer = get_ASTROMER(num_layers=conf['layers'],
                            d_model=conf['head_dim'],
                            num_heads=conf['heads'],
                            dff=conf['dff'],
                            base=conf['base'],
                            dropout=conf['dropout'],
                            maxlen=conf['max_obs'],
                            use_leak=conf['use_leak'],
                            no_train=conf['no_train'])

    # Loading pretrained weights
    weights_path = '{}/weights'.format(opt.w)
    astromer.load_weights(weights_path)
    astromer.trainable = False
    encoder = astromer.get_layer('encoder')

    train_batches = load_records(os.path.join(opt.data, 'train'),
                                 opt.batch_size,
                                 max_obs=conf['max_obs'],
                                 msk_frac=0.,
                                 rnd_frac=0.,
                                 same_frac=0.,
                                 repeat=opt.repeat,
                                 is_train=True)

    valid_batches = load_records(os.path.join(opt.data, 'val'),
                                 opt.batch_size,
                                 max_obs=conf['max_obs'],
                                 msk_frac=0.,
                                 rnd_frac=0.,
                                 same_frac=0.,
                                 repeat=opt.repeat,
                                 is_train=True)

    try:
        test_batches = load_records(os.path.join(opt.data, 'test'),
                                    opt.batch_size,
                                    max_obs=conf['max_obs'],
                                    msk_frac=0.,
                                    rnd_frac=0.,
                                    same_frac=0.,
                                    repeat=1,
                                    is_train=False)
        using_test = True
    except:
        using_test = False
        print('NO TEST')

    os.makedirs(os.path.join(opt.p, 'train'), exist_ok=True)
    for i, batch in enumerate(train_batches):
        with h5py.File(os.path.join(opt.p,'train','batch_{}.h5'.format(i)), 'w') as hf:
            att = encoder(batch)
            lenghts = tf.reduce_sum(1.-batch['mask_in'], 1)[...,0]
            hf.create_dataset('lengths', data=lenghts.numpy())
            hf.create_dataset('embs', data=att.numpy())
            hf.create_dataset('labels', data=batch['label'].numpy())
            hf.create_dataset('oids', data=batch['lcid'].numpy().astype('S'))

    copyfile(os.path.join(opt.data, 'test_objs.csv'),
             os.path.join(opt.p, 'test_objs.csv'))

    os.makedirs(os.path.join(opt.p, 'val'), exist_ok=True)
    for i, batch in enumerate(valid_batches):
        with h5py.File(os.path.join(opt.p,'val','batch_{}.h5'.format(i)), 'w') as hf:
            att = encoder(batch)
            lenghts = tf.reduce_sum(1.-batch['mask_in'], 1)[...,0]
            hf.create_dataset('lengths', data=lenghts.numpy())
            hf.create_dataset('embs', data=att.numpy())
            hf.create_dataset('labels', data=batch['label'].numpy())
            hf.create_dataset('oids', data=batch['lcid'].numpy().astype('S'))

    if using_test:
        os.makedirs(os.path.join(opt.p, 'test'), exist_ok=True)
        for i, batch in enumerate(test_batches):
            with h5py.File(os.path.join(opt.p,'test','batch_{}.h5'.format(i)), 'w') as hf:
                att = encoder(batch)
                lenghts = tf.reduce_sum(1.-batch['mask_in'], 1)[...,0]
                hf.create_dataset('lengths', data=lenghts.numpy())
                hf.create_dataset('embs', data=att.numpy())
                hf.create_dataset('labels', data=batch['label'].numpy())
                hf.create_dataset('oids', data=batch['lcid'].numpy().astype('S'))


    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))


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
    parser.add_argument('--repeat', default=1, type=int,
                        help='repeat')

    opt = parser.parse_args()
    run(opt)
