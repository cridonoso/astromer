
import tensorflow as tf
import argparse
import sys
import os

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.models.astromer_skip import get_ASTROMER as ASTROMER_SKIP
from src.models.astromer_gap import get_ASTROMER as ASTROMER_GAP
from src.models.astromer_nsp import get_ASTROMER as ASTROMER_NSP

from src.data import get_loader


from datetime import datetime


def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    ROOT = './presentation/experiments/astromer_2/'
    trial = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPDIR = os.path.join(ROOT, 'results', opt.exp_name, trial, 'pretraining')

    # ========== DATA ========================================
    train_loader = get_loader(os.path.join(opt.data, 'train'),
                              batch_size=5 if opt.debug else opt.bs,
                              window_size=opt.window_size,
                              probed_frac=opt.probed,
                              random_frac=opt.rs,
                              nsp_prob=opt.nsp_prob,
                              sampling=True,
                              shuffle=True,
                              repeat=4,
                              aversion=opt.encoder_mode)

    valid_loader = get_loader(os.path.join(opt.data, 'val'),
                              batch_size=5 if opt.debug else opt.bs,
                              window_size=opt.window_size,
                              probed_frac=opt.probed,
                              random_frac=opt.rs,
                              nsp_prob=opt.nsp_prob,
                              sampling=True,
                              shuffle=False,
                              repeat=1,
                              aversion=opt.encoder_mode)

    # ======= MODEL ========================================
    if opt.encoder_mode == 'nsp':
        model = ASTROMER_NSP(num_layers=opt.num_layers,
                            num_heads=opt.num_heads,
                            head_dim=opt.head_dim,
                            mixer_size=opt.mixer,
                            dropout=opt.dropout,
                            pe_base=opt.pe_base,
                            pe_dim=opt.pe_dim,
                            pe_c=opt.pe_exp,
                            window_size=opt.window_size)

    if opt.encoder_mode == 'gap':
        model = ASTROMER_GAP(num_layers=opt.num_layers,
                            num_heads=opt.num_heads,
                            head_dim=opt.head_dim,
                            mixer_size=opt.mixer,
                            dropout=opt.dropout,
                            pe_base=opt.pe_base,
                            pe_dim=opt.pe_dim,
                            pe_c=opt.pe_exp,
                            window_size=opt.window_size)

    if opt.encoder_mode == 'skip':
        model = ASTROMER_SKIP(num_layers=opt.num_layers,
                            num_heads=opt.num_heads,
                            head_dim=opt.head_dim,
                            mixer_size=opt.mixer,
                            dropout=opt.dropout,
                            pe_base=opt.pe_base,
                            pe_dim=opt.pe_dim,
                            pe_c=opt.pe_exp,
                            window_size=opt.window_size)

    if opt.checkpoint != '-1':
        print('[INFO] Restoring previous training')
        model.load_weights(os.path.join(opt.checkpoint, 'weights'))
        
    # ============================================================
    cbks = [TensorBoard(log_dir=os.path.join(EXPDIR, 'tensorboard')),
            EarlyStopping(monitor='val_loss', patience=opt.patience),
            ModelCheckpoint(filepath=os.path.join(EXPDIR, 'weights'),
                save_weights_only=True,
                save_best_only=True,
                save_freq='epoch',
                verbose=1)]


    model.compile(optimizer=Adam(1e-3))

    model.fit(train_loader, 
              epochs=2 if opt.debug else opt.num_epochs, 
              validation_data=valid_loader,
              callbacks=cbks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', default='pretrain', type=str,
                    help='Project name')
    parser.add_argument('--data', default='./data/records/macho', type=str,
                    help='Data folder where tf.record files are located')
    parser.add_argument('--checkpoint', default='-1', type=str,
                        help='Restore training by using checkpoints. This is the route to the checkpoint folder.')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')

    parser.add_argument('--encoder-mode', default='skip', type=str,
                        help='skip/nsp')
    parser.add_argument('--num-layers', default=2, type=int,
                        help='Number of Attention Layers')
    parser.add_argument('--num-heads', default=4, type=int,
                        help='Number of heads within the attention layer')
    parser.add_argument('--head-dim', default=64, type=int,
                        help='Head dimension')
    parser.add_argument('--pe-dim', default=256, type=int,
                        help='Positional encoder size - i.e., Number of frequencies')
    parser.add_argument('--pe-base', default=1000, type=int,
                        help='Positional encoder base')
    parser.add_argument('--pe-exp', default=2, type=int,
                        help='Positional encoder exponent')
    parser.add_argument('--mixer', default=128, type=int,
                        help='Units to be used on the hidden layer of a feed-forward network that combines head outputs within an attention layer')
    parser.add_argument('--dropout', default=0., type=float,
                        help='Dropout to use on the output of each attention layer (before mixer layer)')
    
    parser.add_argument('--rmse-factor', default=0.5, type=float,
                        help='RMSE weight factor. The loss function will be loss = rmse_factor*rmse + (1 - rmse_factor)*bce')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--bs', default=2500, type=int,
                        help='Batch size')
    parser.add_argument('--patience', default=20, type=int,
                        help='Earlystopping threshold in number of epochs')
    parser.add_argument('--num_epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--window-size', default=200, type=int,
                        help='windows size of the PSFs')\

    parser.add_argument('--probed', default=0.5, type=float,
                        help='Probed percentage')
    parser.add_argument('--rs', default=0.2, type=float,
                        help='Probed fraction to be randomized or unmasked')
    parser.add_argument('--nsp-prob', default=0.5, type=float,
                        help='Next segment prediction probability. The probability of randomize half of the light curve')

    opt = parser.parse_args()        
    run(opt)
