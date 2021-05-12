import tensorflow as tf
import argparse
import logging
import json
import time
import os

from core.data  import load_records
from core.astromer import get_ASTROMER, get_FINETUNING, train

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

def run(opt):
    # Loading Pretrained model hyperparameters
    conf_file = os.path.join(opt.pretrained, 'conf.json')
    with open(conf_file, 'r') as handle:
        conf = json.load(handle)

    # Loading data
    train_batches = load_records(os.path.join(conf['data'], 'train'),
                                 conf['batch_size'],
                                 input_len=conf['max_obs'],
                                 repeat=opt.repeat,
                                 balanced=True,
                                 finetuning=conf['finetuning'])
    valid_batches = load_records(os.path.join(conf['data'], 'val'),
                                 conf['batch_size'],
                                 input_len=conf['max_obs'],
                                 repeat=opt.repeat,
                                 balanced=True,
                                 finetuning=conf['finetuning'])
    # Num classes
    uniques, _ = tf.unique([b['label']for b in train_batches.unbatch()])
    num_cls = uniques.shape[0]

    # get_model
    astromer = get_ASTROMER(num_layers=conf['layers'],
                            d_model=conf['head_dim'],
                            num_heads=conf['heads'],
                            dff=conf['dff'],
                            base=conf['base'],
                            dropout=conf['dropout'],
                            maxlen=conf['max_obs'])
    weights_path = '{}/weights'.format(conf['p'])
    astromer.load_weights(weights_path)

    finetuning = get_FINETUNING(astromer, num_cls=num_cls)

    opt.p = os.path.join(conf['p'], 'finetuning')
    os.makedirs(opt.p, exist_ok=True)
    tf.keras.utils.plot_model(finetuning,
                              to_file='{}/model.png'.format(opt.p),
                              show_shapes=True)

    # Training ASTROMER
    train(finetuning, train_batches, valid_batches,
          patience=opt.patience,
          exp_path=opt.p,
          epochs=opt.epochs,
          finetuning=opt.finetuning,
          use_random=False,
          num_cls=num_cls,
          verbose=0)

    # Save hyperparameters
    opt_dic = vars(opt)
    for key, value in conf.items():
        if key not in opt_dic.keys():
            opt_dic[key] = value

    conf_file = os.path.join(opt.p, 'conf.json')
    with open(conf_file, 'w') as json_file:
        json.dump(opt_dic, json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DATA
    parser.add_argument('--max-obs', default=100, type=int,
                    help='Max number of observations')
    # TRAINING PAREMETERS
    parser.add_argument('--data', default='./data/records/macho', type=str,
                        help='Dataset folder containing the records files')
    parser.add_argument('--p', default="./experiments/debug", type=str,
                        help='Proyect path. Here will be stored weights and metrics')
    parser.add_argument('--batch-size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=2000, type=int,
                        help='Number of epochs')
    parser.add_argument('--patience', default=20, type=int,
                        help='batch size')
    parser.add_argument('--finetuning',default=True, action='store_true',
                        help='Finetune a pretrained model')
    parser.add_argument('--repeat', default=5, type=int,
                        help='number of times to repeat the training and validation dataset')
    # ASTROMER HIPERPARAMETERS
    parser.add_argument('--pretrained', type=str,
                        help='pretrained model directory')


    opt = parser.parse_args()
    run(opt)
