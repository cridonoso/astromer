import tensorflow as tf
import argparse
import sys
import os

from src.models.astromer_1 import get_ASTROMER, train_step, test_step

from src.training.utils import train
from src.data import load_data
from datetime import datetime



def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    ROOT = './presentation/experiments/astromer_1/'
    trial = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPDIR = os.path.join(ROOT, 'results', opt.exp_name, trial, 'pretraining')
    os.makedirs(EXPDIR, exist_ok=True)

    # ========== DATA ========================================
    train_loader = load_data(dataset=os.path.join(opt.data, 'train'), 
                             batch_size=5 if opt.debug else opt.bs, 
                             probed=opt.probed,
                             random_same=opt.rs,  
                             window_size=opt.window_size, 
                             off_nsp=True,  
                             repeat=4, 
                             sampling=True)
    valid_loader = load_data(dataset=os.path.join(opt.data, 'val'), 
                             batch_size=5 if opt.debug else opt.bs, 
                             probed=opt.probed,  
                             random_same=opt.rs,
                             window_size=opt.window_size, 
                             off_nsp=True,  
                             repeat=1, 
                             sampling=True)
    test_loader = load_data(dataset=os.path.join(opt.data, 'test'), 
                             batch_size=5 if opt.debug else opt.bs, 
                             probed=opt.probed,  
                             random_same=opt.rs,
                             window_size=opt.window_size, 
                             off_nsp=True, 
                             repeat=1, 
                             sampling=True)

    # ======= MODEL ========================================
    model = get_ASTROMER(num_layers=opt.num_layers,
                        num_heads=opt.num_heads,
                        head_dim=opt.head_dim,
                        mixer_size=opt.mixer,
                        dropout=opt.dropout,
                        pe_base=opt.pe_base,
                        pe_dim=opt.pe_dim,
                        pe_c=opt.pe_exp,
                        window_size=opt.window_size,
                        encoder_mode=opt.encoder_mode,
                        average_layers=opt.avg_layers,
                        mask_format=opt.mask_format)

    # ============================================================
    if opt.checkpoint != '-1':
        print('[INFO] Restoring previous training')
        model.load_weights(os.path.join(opt.checkpoint, 'weights', 'weights'))
        
    model = train(model,
              train_loader, 
              valid_loader, 
              num_epochs=opt.num_epochs, 
              lr=opt.lr, 
              test_loader=test_loader,
              project_path=EXPDIR,
              debug=opt.debug,
              patience=opt.patience,
              train_step_fn=train_step,
              test_step_fn=test_step,
              argparse_dict=opt.__dict__)


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

	parser.add_argument('--encoder-mode', default='normal', type=str,
						help='normal - conditioned')
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
	parser.add_argument('--avg-layers', action='store_true', help='If averaging outputs of the attention layers to form the final embedding. There is no avg if layers=1 ')
	parser.add_argument('--mask-format', default='first', type=str,
						help='first - zero')
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


	opt = parser.parse_args()        
	run(opt)
