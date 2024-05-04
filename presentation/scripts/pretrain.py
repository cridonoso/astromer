import tensorflow as tf
import argparse
import toml
import sys
import os


from src.models.astromer_0 import get_ASTROMER as get_Bugstromer
from src.models.astromer_1 import get_ASTROMER as get_Base
from src.models.astromer_nsp import get_ASTROMER as get_NSP
from src.models.astromer_skip import get_ASTROMER as get_Skip

from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from.src.training.utils import train
from src.training.scheduler import CustomSchedule

from presentation.pipelines.steps.model_design import build_model
from presentation.pipelines.steps.load_data import build_loader

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    ROOT = './presentation/'
    trial = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPDIR = os.path.join(ROOT, 'results', opt.exp_name, trial, 'pretraining')
    os.makedirs(EXPDIR, exist_ok=True)

    # ========== DATA ========================================
    loaders = build_loader(opt)

    # ======= MODEL ========================================
    model = get_model(opt)

    # ============================================================
    if opt.checkpoint != '-1':
        print('[INFO] Restoring previous training')
        model.load_weights(os.path.join(opt.checkpoint, 'weights'))
        
    with open(os.path.join(EXPDIR, 'config.toml'), 'w') as f:
        toml.dump(opt.__dict__, f)

    train(model,
              train_dataset,
              valid_dataset,
              patience=20,
              exp_path='./experiments/test',
              epochs=1,
              finetuning=False,
              use_random=True,
              num_cls=2,
              lr=1e-3,
              verbose=1)


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
    parser.add_argument('--scheduler', action='store_true', help='Use Custom Scheduler during training')

    parser.add_argument('--arch', default='base', type=str,
                        help='Astromer architecture: "base" (paper), "nsp", or "skip"')

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
    parser.add_argument('--m-alpha', default=1., type=float,
                        help='Alpha used within mask self-attention')
    parser.add_argument('--mask-format', default=None, type=str,
                        help='mask on Query and Key tokens (QK) or Query tokens only (Q)')
    parser.add_argument('--use-leak', action='store_true', help='Use Custom Scheduler during training')
    parser.add_argument('--no-cache', action='store_true', help='no cache dataset')
    
    parser.add_argument('--correct-loss', action='store_true', help='Use error bars to weigh loss')
    parser.add_argument('--loss-format', default='rmse', type=str,
                        help='what consider during loss: rmse - rmse+p - p')
    parser.add_argument('--norm', default='zero-mean', type=str,
                        help='normalization: zero-mean - random-mean')
    parser.add_argument('--temperature', default=0., type=float,
                        help='Temperature used within the softmax argument')
    # =========================================================
    parser.add_argument('--repeat', default=1, type=int,
                        help='repeat data')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--bs', default=2500, type=int,
                        help='Batch size')
    parser.add_argument('--patience', default=20, type=int,
                        help='Earlystopping threshold in number of epochs')
    parser.add_argument('--num_epochs', default=10000, type=int,
                        help='Number of epochs')
    parser.add_argument('--window-size', default=200, type=int,
                        help='windows size of the PSFs')
    # ==========================================================
    parser.add_argument('--probed', default=0.2, type=float,
                        help='Probed percentage')
    parser.add_argument('--rs', default=0.2, type=float,
                        help='Probed fraction to be randomized or unmasked')
    parser.add_argument('--same', default=None, type=float,
                        help='Fraction to make visible during masked-self attention while evaluating during loss')
    # ONLY NSP =================================================
    parser.add_argument('--nsp-prob', default=0.5, type=float,
                        help='Probability of randomize second segment in NSP pretraining task')


    opt = parser.parse_args()        
    run(opt)
