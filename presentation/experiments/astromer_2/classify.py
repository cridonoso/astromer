import tensorflow as tf
import pandas as pd
import argparse
import toml
import sys
import os

from presentation.experiments.utils import train_classifier

from src.models.astromer_skip import get_ASTROMER as ASTROMER_SKIP, build_input as build_input_skip
from src.models.astromer_gap import get_ASTROMER as ASTROMER_GAP, build_input
from src.models.astromer_nsp import get_ASTROMER as ASTROMER_NSP, build_input

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.data import get_loader
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
        
    # ======= MODEL ========================================
    if model_config['encoder_mode'] == 'nsp':
        astromer = ASTROMER_NSP(num_layers=model_config['num_layers'],
                            num_heads=model_config['num_heads'],
                            head_dim=model_config['head_dim'],
                            mixer_size=model_config['mixer'],
                            dropout=model_config['dropout'],
                            pe_base=model_config['pe_base'],
                            pe_dim=model_config['pe_dim'],
                            pe_c=model_config['pe_exp'],
                            window_size=model_config['window_size'])
        inp_placeholder = build_input(model_config['window_size'])

    if model_config['encoder_mode'] == 'gap':
        astromer = ASTROMER_GAP(num_layers=model_config['num_layers'],
                            num_heads=model_config['num_heads'],
                            head_dim=model_config['head_dim'],
                            mixer_size=model_config['mixer'],
                            dropout=model_config['dropout'],
                            pe_base=model_config['pe_base'],
                            pe_dim=model_config['pe_dim'],
                            pe_c=model_config['pe_exp'],
                            window_size=model_config['window_size'])
        inp_placeholder = build_input(model_config['window_size'])

    if model_config['encoder_mode'] == 'skip':
        astromer = ASTROMER_SKIP(num_layers=model_config['num_layers'],
                            num_heads=model_config['num_heads'],
                            head_dim=model_config['head_dim'],
                            mixer_size=model_config['mixer'],
                            dropout=model_config['dropout'],
                            pe_base=model_config['pe_base'],
                            pe_dim=model_config['pe_dim'],
                            pe_c=model_config['pe_exp'],
                            window_size=model_config['window_size'])
        inp_placeholder = build_input_skip(model_config['window_size'])

    astromer.load_weights(os.path.join(opt.ft_folder, 'weights')).expect_partial()
    print('[INFO] Weights loaded')

    # ====================================================================================
    # =============== DOWNSTREAM TASK  ===================================================
    # ====================================================================================
    num_cls = pd.read_csv(os.path.join(opt.data, 'objects.csv')).shape[0]
    # ========== DATA ========================================
    train_loader = get_loader(os.path.join(opt.data, 'train'),
                              batch_size=5 if opt.debug else opt.bs,
                              window_size=model_config['window_size'],
                              probed_frac=1.,
                              random_frac=0.,
                              sampling=False,
                              shuffle=True,
                              repeat=1,
                              aversion=model_config['encoder_mode'],
                              num_cls=num_cls)

    valid_loader = get_loader(os.path.join(opt.data, 'val'),
                              batch_size=5 if opt.debug else opt.bs,
                              window_size=model_config['window_size'],
                              probed_frac=1.,
                              random_frac=0.,
                              sampling=False,
                              shuffle=False,
                              repeat=1,
                              aversion=model_config['encoder_mode'],
                              num_cls=num_cls)

    test_loader = get_loader(os.path.join(opt.data, 'test'),
                              batch_size=5 if opt.debug else opt.bs,
                              window_size=model_config['window_size'],
                              probed_frac=1.,
                              random_frac=0.,
                              sampling=False,
                              shuffle=False,
                              repeat=1,
                              aversion=model_config['encoder_mode'],
                              num_cls=num_cls)

    if opt.debug:
        train_loader = train_loader.take(1)
        valid_loader = valid_loader.take(1)
        test_loader  = test_loader.take(1)

    
    encoder = astromer.get_layer('encoder')
    encoder.trainable = opt.train_astromer
    embedding = encoder(inp_placeholder)
    
    if 'att' in opt.clf_name and model_config['encoder_mode'] == 'skip':
        print('[INFO] Using SKIP')
        embedding = embedding*(1.-inp_placeholder['att_mask'])
        embedding = tf.math.divide_no_nan(tf.reduce_sum(embedding, axis=1), 
                                  tf.reduce_sum(1.-inp_placeholder['att_mask'], axis=1))

    if 'cls' in opt.clf_name and model_config['encoder_mode'] != 'skip':
        print('[INFO] Using CLS tokens')
        embedding = tf.slice(embedding, [0, 0, 0], [-1, 1,-1], name='slice_cls')
        embedding = tf.squeeze(embedding, axis=1)

    if 'att' in opt.clf_name and model_config['encoder_mode'] != 'skip':
        print('[INFO] Using OBS tokens')
        embedding = tf.slice(embedding, [0, 1, 0], [-1, 1,-1], name='slice_att')
        embedding = embedding*(1.-inp_placeholder['att_mask'])
        embedding = tf.math.divide_no_nan(tf.reduce_sum(embedding, axis=1), 
                                          tf.reduce_sum(1.-inp_placeholder['att_mask'], axis=1))

    if 'all' in opt.clf_name and model_config['encoder_mode'] != 'skip':
        print('[INFO] Using ALL tokens')
        cls_token  = tf.slice(embedding, [0, 0, 0], [-1, 1,-1], name='slice_cls')
        cls_token = tf.squeeze(cls_token, axis=1)
        att_tokens = tf.slice(embedding, [0, 1, 0], [-1, 1,-1], name='slice_att')
        att_tokens = att_tokens*(1.-inp_placeholder['att_mask'])
        att_tokens = tf.math.divide_no_nan(tf.reduce_sum(att_tokens, axis=1), 
                                          tf.reduce_sum(1.-inp_placeholder['att_mask'], axis=1))
        embedding = tf.concat([att_tokens, cls_token], axis=-1)

    summary_clf = train_classifier(embedding,
                                   inp_placeholder=inp_placeholder,
                                   train_loader=train_loader,
                                   valid_loader=valid_loader, 
                                   test_loader=test_loader,
                                   num_cls=num_cls, 
                                   project_path=opt.clf_folder,
                                   clf_name=opt.clf_name,
                                   debug=opt.debug)
    
    with open(os.path.join(opt.clf_folder, 'config.toml'), 'w') as f:
        toml.dump(model_config, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='-1', type=str, help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--data', default='./data/records/alcock/fold_0/alcock_20', type=str, help='Data folder where tf.record files are located')
    parser.add_argument('--pt-folder', default='./presentation/experiments/astromer_2/results/epoch_3090/', type=str, help='Pretraining folder')
    parser.add_argument('--ft-folder', default='./presentation/experiments/astromer_2/results/epoch_3090/finetuning/alcock/fold_0/alcock_20', type=str, help='Finetuning folder')
    parser.add_argument('--clf-folder', default='./presentation/experiments/astromer_2/results/epoch_3090/classification/', type=str, help='Classification folder')
    
    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')
    parser.add_argument('--bs', default=256, type=int, help='Batch size')
    parser.add_argument('--patience', default=20, type=int, help='Earlystopping threshold in number of epochs')
    parser.add_argument('--num_epochs', default=10000, type=int, help='Number of epochs')

    parser.add_argument('--train-astromer', action='store_true', help='If train astromer when classifying')
    parser.add_argument('--clf-name', default='att_linear', type=str, help='classifier name')

    opt = parser.parse_args()        
    run(opt)