import tensorflow as tf
import argparse
import logging
import json
import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from core.utils import get_folder_name, dict_to_json
from core.data  import pretraining_records
from core.models import get_ASTROMER

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings


def run(opt):
    # Get model
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    # Make sure we don't overwrite a previous training
    opt.p = get_folder_name(opt.p, prefix='')

    # Creating (--p)roject directory
    os.makedirs(opt.p, exist_ok=True)

    # Check if there is pretraining weights to be loaded
    varsdic = vars(opt)
    if opt.w !='':
        print('[INFO] Pretrained model detected! - Finetuning...')
        conf_file = os.path.join(opt.w, 'conf.json')
        with open(conf_file, 'r') as handle:
            conf = json.load(handle)
        # Changing "opt" hyperparameters
        for key in conf.keys():
            # Don't include parameters exclusive to this training
            if key in ['batch_size', 'p', 'repeat', 'data', 'patience',
                       'msk_frac', 'rnd_frac', 'same_frac']:
                continue
            varsdic[key] = conf[key]

    # Saving new updated conf_file
    conf_file = os.path.join(varsdic['p'], 'conf.json')
    dict_to_json(varsdic, conf_file)

    # Instance the model
    astromer = get_ASTROMER(num_layers=varsdic['layers'],
                            d_model=varsdic['head_dim'],
                            num_heads=varsdic['heads'],
                            dff=varsdic['dff'],
                            base=varsdic['base'],
                            rate=varsdic['dropout'],
                            maxlen=varsdic['max_obs'])
    # Load weights if exist
    if varsdic['w'] != '':
        astromer.load_weights(os.path.join(varsdic['w'], 'weights'))

    # Compile model
    # Losses and metrics have been already included in core.models.zero
    optimizer = tf.keras.optimizers.Adam(varsdic['lr'],
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    astromer.compile(optimizer=optimizer)

    # Loading and formating data
    train_batches = pretraining_records(os.path.join(varsdic['data'], 'train'),
                                        varsdic['batch_size'],
                                        max_obs=varsdic['max_obs'],
                                        shuffle=True,
                                        sampling=True,
                                        msk_frac=varsdic['msk_frac'],
                                        rnd_frac=varsdic['rnd_frac'],
                                        same_frac=varsdic['same_frac'])
    valid_batches = pretraining_records(os.path.join(varsdic['data'], 'val'),
                                        varsdic['batch_size'],
                                        max_obs=varsdic['max_obs'],
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=varsdic['msk_frac'],
                                        rnd_frac=varsdic['rnd_frac'],
                                        same_frac=varsdic['same_frac'])

    # Setting up callbacks
    callbacks = [
            ModelCheckpoint(
                    filepath=os.path.join(varsdic['p'], 'weights'),
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True),
            EarlyStopping(monitor ='val_loss',
                          mode = 'min',
                          patience = varsdic['patience'],
                          restore_best_weights=True),
            TensorBoard(
                    log_dir = os.path.join(varsdic['p'], 'logs'),
                    histogram_freq=1,
                    write_graph=True)
    ]

    # Training
    astromer.fit(train_batches,
                 epochs=varsdic['epochs'],
                 validation_data=valid_batches,
                 callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')

    parser.add_argument('--msk-frac', default=0.5, type=float,
                        help='[MASKED] fraction')
    parser.add_argument('--rnd-frac', default=0.2, type=float,
                        help='Fraction of [MASKED] to be replaced by random values')
    parser.add_argument('--same-frac', default=0.2, type=float,
                        help='Fraction of [MASKED] to be replaced by same values')

    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--w', default="", type=str,
                        help='[OPTIONAL] pre-training weights')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=40, type=int,
                        help='batch size')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU to use')

    # ASTROMER HIPERPARAMETERS
    parser.add_argument('--layers', default=2, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--heads', default=4, type=int,
                        help='Number of self-attention heads')
    parser.add_argument('--head-dim', default=256, type=int,
                        help='Head-attention Dimensionality ')
    parser.add_argument('--dff', default=128, type=int,
                        help='Dimensionality of the middle  dense layer at the end of the encoder')
    parser.add_argument('--dropout', default=0.1 , type=float,
                        help='dropout_rate for the encoder')
    parser.add_argument('--base', default=1000, type=int,
                        help='base of embedding')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')

    opt = parser.parse_args()
    run(opt)
