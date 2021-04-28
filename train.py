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

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def train(opt):
    # Loading data
    train_batches = load_records(os.path.join(opt.data, 'train'),
                                 opt.batch_size,
                                 input_len=opt.max_obs)
    valid_batches = load_records(os.path.join(opt.data, 'val'),
                                 opt.batch_size,
                                 input_len=opt.max_obs)
    test_batches = load_records(os.path.join(opt.data, 'test'),
                                 opt.batch_size,
                                 input_len=opt.max_obs)

    # Optimizer
    learning_rate = 1e-3#CustomSchedule(opt.head_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    # Model Instance
    transformer = ASTROMER(num_layers=opt.layers,
                           d_model=opt.head_dim,
                           num_heads=opt.heads,
                           dff=opt.dff,
                           rate=opt.dropout,
                           base=opt.base,
                           mask_frac=0.15)
    # Compile
    transformer.compile(optimizer=optimizer,
                        loss=ASTROMERLoss(),
                        metrics=[CustomMSE(), CustomBCE(), CustomACC()])
    # Create graph
    transformer.model(opt.batch_size).summary()


    # Training
    transformer.fit(train_batches,
                    epochs=opt.epochs,
                    verbose=1,
                    validation_data=valid_batches,
                    callbacks=get_callbacks(opt.p))
    # Testing
    metrics = transformer.evaluate(test_batches)

    # Saving metrics and setup file
    os.makedirs(os.path.join(opt.p, 'test'), exist_ok=True)
    test_file = os.path.join(opt.p, 'test/test_metrics.json')
    with open(test_file, 'w') as json_file:
        json.dump({'loss': metrics[0],
                   'rmse':metrics[1],
                   'accuracy':metrics[2]}, json_file, indent=4)

    conf_file = os.path.join(opt.p, 'conf.json')
    with open(conf_file, 'w') as json_file:
        json.dump(vars(opt), json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=100, type=int,
                    help='Max number of observations')
    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./experiments/macho", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=2000, type=int,
                        help='Number of epochs')
    # ASTROMER HIPERPARAMETERS
    parser.add_argument('--layers', default=2, type=int,
                        help='Number of encoder layers')
    parser.add_argument('--heads', default=4, type=int,
                        help='Number of self-attention heads')
    parser.add_argument('--head-dim', default=812, type=int,
                        help='Head-attention Dimensionality ')
    parser.add_argument('--dff', default=1024, type=int,
                        help='Dimensionality of the middle  dense layer at the end of the encoder')
    parser.add_argument('--dropout', default=0.1 , type=float,
                        help='dropout_rate for the encoder')
    parser.add_argument('--base', default=10000, type=int,
                        help='base of embedding')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='optimizer initial learning rate')

    opt = parser.parse_args()
    opt.head_dim = (opt.max_obs + 3)*opt.heads
    train(opt)
