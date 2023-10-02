import tensorflow as tf
import argparse
import sys
import os

from src.models.astromer_1 import get_ASTROMER, train_step, test_step

from src.training.utils import train
from src.data import load_data,load_data_astrospec
from datetime import datetime



def run(opt):
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

	ROOT = './presentation/experiments/astromer_1/astrospec/'
	trial = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	EXPDIR = os.path.join(ROOT, 'results', opt.exp_name, trial, 'pretraining')
	os.makedirs(EXPDIR, exist_ok=True)

	# ========== DATA ========================================
	train_loader = load_data_astrospec(dataset=os.path.join(opt.data, 'train'), 
							 batch_size=5 if opt.debug else opt.bs, 
							 no_of_observations=opt.no_of_observations,
							 non_exclusive_fraction=opt.non_exlu_frac,
							 exclusive_fraction=opt.exlu_frac,
							 iqr_threshold=opt.iqr_thres,
							 random_fraction=opt.random_frac,
							 same_fraction=opt.same_frac,
							 off_nsp=True,  
							 repeat=4, 
							 sampling=True)
	valid_loader = load_data_astrospec(dataset=os.path.join(opt.data, 'val'), 
							 batch_size=5 if opt.debug else opt.bs, 
							 no_of_observations=opt.no_of_observations,
							 non_exclusive_fraction=opt.non_exlu_frac,
							 exclusive_fraction=opt.exlu_frac,
							 iqr_threshold=opt.iqr_thres,
							 random_fraction=opt.random_frac,
							 same_fraction=opt.same_frac,
							 off_nsp=True,  
							 repeat=1, 
							 sampling=True)
	test_loader = load_data_astrospec(dataset=os.path.join(opt.data, 'test'), 
							 batch_size=5 if opt.debug else opt.bs, 
							 no_of_observations=opt.no_of_observations,
							 non_exclusive_fraction=opt.non_exlu_frac,
							 exclusive_fraction=opt.exlu_frac,
							 iqr_threshold=opt.iqr_thres,
							 random_fraction=opt.random_frac,
							 same_fraction=opt.same_frac,
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
						window_size=opt.no_of_observations,
						encoder_mode=opt.encoder_mode,
						astrospec_skip=opt.astrospec_skip,
						average_layers=opt.avg_layers)

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
	parser.add_argument('--data', default='./data/records/macho_clean', type=str,
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
	parser.add_argument('--mixer', default=256, type=int,
						help='Units to be used on the hidden layer of a feed-forward network that combines head outputs within an attention layer')
	parser.add_argument('--dropout', default=0.1, type=float,
						help='Dropout to use on the output of each attention layer (before mixer layer)')
	parser.add_argument('--avg-layers', action='store_true', help='If averaging outputs of the attention layers to form the final embedding. There is no avg if layers=1 ')

	parser.add_argument('--lr', default=1e-5, type=float,
						help='learning rate')
	parser.add_argument('--bs', default=2500, type=int,
						help='Batch size')
	parser.add_argument('--patience', default=20, type=int,
						help='Earlystopping threshold in number of epochs')
	parser.add_argument('--num_epochs', default=10000, type=int,
						help='Number of epochs')
	parser.add_argument('--no-of-observations', default=200, type=int,
						help='no of flux-wavelength pairs per object')\

	parser.add_argument('--same-frac', default=0.2, type=float,
						help='Fraction of masked observations to be unmasked and replaced with same values')
	parser.add_argument('--random-frac', default=0.2, type=float,
						help='Fraction of masked observations to be unmasked and replaced with random values')
	parser.add_argument('--exlu-frac', default=0.6, type=float,
						help='Fraction of observations to be masked in the exclusive zone')
	parser.add_argument('--non-exlu-frac', default=0.4, type=float,
						help='Fraction of observations to be masked in the non exclusive zone')
	parser.add_argument('--iqr-thres', default=1.2, type=float,
						help='Threshold factor for finding exclusive and non exclusive zone')
	parser.add_argument('--astrospec_skip', default=False, type=bool,
						help='Flag which indicates whether to use skip connection in encoder- only for astrospec')
	


	opt = parser.parse_args()        
	run(opt)
