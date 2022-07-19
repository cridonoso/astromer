import tensorflow as tf
import argparse
import json
import os

from core.training.scheduler import CustomSchedule
from core.astromer import ASTROMER
from core.data  import pretraining_pipeline
from core.training.callbacks import get_callbacks
from core.utils import dict_to_json
from tensorflow.keras.callbacks import ModelCheckpoint

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu

    train_ds = pretraining_pipeline(os.path.join(opt.data, 'train'),
                                    batch_size=opt.batch_size,
                                    max_obs=opt.max_obs,
                                    msk_frac=opt.msk_frac,
                                    rnd_frac=opt.rnd_frac,
                                    same_frac=opt.same_frac,
                                    shuffle=True,
                                    sampling=True,
                                    )
    val_ds   = pretraining_pipeline(os.path.join(opt.data, 'val'),
                                    batch_size=opt.batch_size,
                                    max_obs=opt.max_obs,
                                    msk_frac=opt.msk_frac,
                                    rnd_frac=opt.rnd_frac,
                                    shuffle=False,
                                    sampling=True,
                                    same_frac=opt.same_frac,
                                    )

    # Initialize model
    model = ASTROMER(num_layers= opt.layers,
                     d_model   = opt.head_dim,
                     num_heads = opt.heads,
                     dff       = opt.dff,
                     base      = opt.base,
                     dropout   = opt.dropout,
                     maxlen    = opt.max_obs)

    if opt.w != '':
        print('[INFO] Loading pre-trained weights')
        model.load_weights(opt.w)

    # Save Hyperparameters
    dict_to_json(opt, opt.p)

    # Defining optimizer with custom scheduler for the learning rate
    # learning_rate = CustomSchedule(opt.head_dim)
    optimizer = tf.keras.optimizers.Adam(opt.lr)

    callbacks = get_callbacks(opt.p, opt.patience)

    # Compile and train
    model.compile(optimizer=optimizer)
    _ = model.fit(train_ds,
                  epochs=opt.epochs,
                  validation_data=val_ds,
                  callbacks=callbacks)

    model.save_weights('./weights/candidate_macho')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--data', default='./data/records/testing/fold_0/testing', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--max-obs', default=200, type=int,
                    help='Max number of observations')
    parser.add_argument('--repeat', default=5, type=int,
                        help='times to repeat the training set')
    parser.add_argument('--msk-frac', default=0.5, type=float,
                        help='[MASKED] fraction')
    parser.add_argument('--rnd-frac', default=0.2, type=float,
                        help='Fraction of [MASKED] to be replaced by random values')
    parser.add_argument('--same-frac', default=0.2, type=float,
                        help='Fraction of [MASKED] to be replaced by same values')

    # TRAINING PAREMETERS
    parser.add_argument('--p', default="./runs/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--w', default="", type=str,
                        help='pre-trained weights')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=1000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=200, type=int,
                        help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate')
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
    parser.add_argument('--cache', default=False, action='store_true',
                    help='Save batches while iterating over datasets')
    opt = parser.parse_args()
    run(opt)
