import tensorflow as tf
import pandas as pd
import argparse
import toml
import sys
import os

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.models.astromer_1 import get_ASTROMER
from src.data.loaders import get_loader

from datetime import datetime


def merge_metrics(**kwargs):
    merged = {}
    for key, value in kwargs.items():
        for subkey, subvalue in value.items():
            merged['{}_{}'.format(key, subkey)] = subvalue
    return merged

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    # ====================================================================================
    # =============== LOADING PRETRAINED MODEL ===========================================
    # ====================================================================================
    with open(os.path.join(opt.pt_folder, 'config.toml'), 'r') as f:
        model_config = toml.load(f)
        
    astromer = get_ASTROMER(num_layers=model_config['num_layers'],
                            num_heads=model_config['num_heads'],
                            head_dim=model_config['head_dim'],
                            mixer_size=model_config['mixer'],
                            dropout=model_config['dropout'],
                            pe_base=model_config['pe_base'],
                            pe_dim=model_config['pe_dim'],
                            pe_c=model_config['pe_exp'],
                            window_size=model_config['window_size'],
                            mask_format=model_config['mask_format'])

    astromer.load_weights(os.path.join(opt.pt_folder, 'weights')).expect_partial()
    print('[INFO] Weights loaded')
    # ====================================================================================
    # =============== FINETUNING MODEL  ==================================================
    # ====================================================================================
    DOWNSTREAM_DATA = os.path.join('./data/records', 
                               opt.subdataset,
                               'fold_'+str(opt.fold), 
                               '{}_{}'.format(opt.subdataset, opt.spc))
    FTWEIGTHS = opt.target_dir 

    train_loader = get_loader(os.path.join(DOWNSTREAM_DATA, 'train'),
                              batch_size=5 if opt.debug else opt.bs,
                              window_size=model_config['window_size'],
                              probed_frac=1. if opt.allvisible else model_config['probed'],
                              random_frac=0. if opt.allvisible else model_config['rs'],
                              sampling=False,
                              shuffle=True,
                              repeat=1,
                              aversion='base')

    valid_loader = get_loader(os.path.join(DOWNSTREAM_DATA, 'val'),
                              batch_size=5 if opt.debug else opt.bs,
                              window_size=model_config['window_size'],
                              probed_frac=1. if opt.allvisible else model_config['probed'],
                              random_frac=0. if opt.allvisible else model_config['rs'],
                              sampling=False,
                              shuffle=False,
                              repeat=1,
                              aversion='base')


    cbks = [TensorBoard(log_dir=os.path.join(FTWEIGTHS, 'tensorboard')),
            EarlyStopping(monitor='val_loss', patience=opt.patience),
            ModelCheckpoint(filepath=os.path.join(FTWEIGTHS, 'weights'),
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1)]

    if model_config['scheduler']:
        print('[INFO] Using Custom Scheduler')
        lr = CustomSchedule(d_model=int(opt.head_dim*opt.num_heads))
    else:
        lr = model_config['lr']
        
    astromer.compile(optimizer=Adam(lr, 
                     beta_1=0.9,
                     beta_2=0.98,
                     epsilon=1e-9,
                     name='astromer_optimizer'))

    astromer.fit(train_loader, 
              epochs=2 if opt.debug else opt.num_epochs, 
              validation_data=valid_loader,
              callbacks=cbks)

    with open(os.path.join(FTWEIGTHS, 'config.toml'), 'w') as f:
        toml.dump(model_config, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='-1', type=str, help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--subdataset', default='alcock', type=str, help='Data folder where tf.record files are located')
    parser.add_argument('--pt-folder', default='./results/pretraining*', type=str, help='pretrained model folder')
    parser.add_argument('--target-dir', default='./results/finetuning/alcock/fold_0/alcock_20', type=str, help='target directory')

    parser.add_argument('--fold', default=0, type=int, help='Fold to use')
    parser.add_argument('--spc', default=20, type=int, help='Samples per class')
    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')
    parser.add_argument('--exp-name', default='finetuning', type=str, help='folder name where logs/weights will be stored')

    parser.add_argument('--allvisible', action='store_true', help='Disable masking task. All observations are visible')
    parser.add_argument('--bs', default=2000, type=int, help='Batch size')
    parser.add_argument('--patience', default=20, type=int, help='Earlystopping threshold in number of epochs')
    parser.add_argument('--num_epochs', default=10000, type=int, help='Number of epochs')
    parser.add_argument('--train-astromer', action='store_true', help='If train astromer when classifying')
    parser.add_argument('--clf-name', default='att_mlp', type=str, help='classifier name')

    opt = parser.parse_args()        
    run(opt)
