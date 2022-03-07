import tensorflow as tf
import argparse
import logging
import json
import time
import os

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from core.data  import load_dataset, pretraining_pipeline
from core.training.metrics import custom_r2
from core.training.losses import custom_rmse
from core.astromer import ASTROMER

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

    train_ds = load_dataset(os.path.join(opt.data, 'train'))
    val_ds   = load_dataset(os.path.join(opt.data, 'val'))

    train_ds = pretraining_pipeline(train_ds,
                                    batch_size=opt.batch_size,
                                    max_obs=opt.max_obs,
                                    msk_frac=opt.msk_frac,
                                    rnd_frac=opt.rnd_frac,
                                    same_frac=opt.same_frac)
    val_ds   = pretraining_pipeline(val_ds,
                                    batch_size=opt.batch_size,
                                    max_obs=opt.max_obs,
                                    msk_frac=opt.msk_frac,
                                    rnd_frac=opt.rnd_frac,
                                    same_frac=opt.same_frac)

    # Initialize model
    model = ASTROMER(num_layers= opt.layers,
                     d_model   = opt.head_dim,
                     num_heads = opt.heads,
                     dff       = opt.dff,
                     base      = opt.base,
                     dropout   = opt.dropout,
                     use_leak  = opt.use_leak,
                     maxlen    = opt.max_obs)

    model.build({'input': [opt.batch_size, opt.max_obs, 1],
                 'mask_in': [opt.batch_size, opt.max_obs, 1],
                 'times': [opt.batch_size, opt.max_obs, 1]})
    if opt.w != '':
        print('[INFO] Loading pre-trained weights')
        model.load_weights(os.path.join(opt.w, 'weights.h5'))

    model.compile(optimizer='adam',
                  loss_rec=custom_rmse,
                  metric_rec=custom_r2)

    ckp_callback = ModelCheckpoint(
                    filepath=os.path.join(opt.p, 'weights.h5'),
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)
    esp_callback = EarlyStopping(monitor ='val_loss',
                                 mode = 'min',
                                 patience = opt.patience,
                                 restore_best_weights=True)
    tsb_callback = TensorBoard(
                    log_dir = os.path.join(opt.p, 'logs'),
                    write_graph=False)

    history = model.fit(train_ds,
                        epochs=opt.epochs,
                        validation_data=val_ds,
                        callbacks=[ckp_callback, esp_callback, tsb_callback])


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
    parser.add_argument('--data', default='./data/records/testing/fold_0/testing', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--w', default="", type=str,
                        help='pre-trained weights')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=1000, type=int,
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

    parser.add_argument('--use-leak', default=False, action='store_true',
                        help='Add the input to the attention vector')

    opt = parser.parse_args()
    run(opt)
