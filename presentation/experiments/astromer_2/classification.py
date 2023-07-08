import tensorflow as tf
import argparse
import os

from src.layers.downstream import AstromerEmbedding
from src.data.loaders import load_light_curves





def run(opt):

	astromer_layer = AstromerEmbedding(pretrain_weights=opt.pre_weights)

	train_batches = load_light_curves(os.path.join(opt.data, 'train'), 
				                      batch_size=opt.bs, 
				                      window_size=opt.ws, 
				                      repeat=1,
				                      cache=True, 
				                      njobs=None)


	for x, y in train_batches:
		x_emb = astromer_layer(x)
		print(x_emb.shape)
		break
		
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre-weights', default='./presentation/experiments/astromer_2/results/nsp_adamw_factor/1_4_64_rmse_0.5/pretraining', type=str,
                    help='Pretrain weights folder')
    parser.add_argument('--data', default='./data/records/atlas/fold_0/atlas_20', type=str,
                    help='Data folder where tf.record files are located')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU to be used. -1 means no GPU will be used')
    parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')



    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--bs', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--patience', default=20, type=int,
                        help='Earlystopping threshold in number of epochs')
    parser.add_argument('--epochs', default=100000, type=int,
                        help='Number of epochs')
    parser.add_argument('--ws', default=200, type=int,
                        help='windows size of the PSFs')

    parser.add_argument('--probed', default=0.5, type=float,
                        help='Probed percentage')
    parser.add_argument('--nsp-prob', default=0.5, type=float,
                        help='Next segment prediction probability. The probability of randomize half of the light curve')
    parser.add_argument('--rmse-factor', default=0.5, type=float,
                        help='RMSE weight factor. The loss function will be loss = rmse_factor*rmse + (1 - rmse_factor)*bce')


    opt = parser.parse_args()        
    run(opt)