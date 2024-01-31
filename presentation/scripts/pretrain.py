import tensorflow as tf
import argparse
import toml
import sys
import os

from src.models.astromer_0 import get_ASTROMER

from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.training.scheduler import CustomSchedule
from src.data.zero import pretraining_pipeline

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    ROOT = './presentation/results/'
    trial = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPDIR = os.path.join(ROOT, 'results', opt.exp_name, trial, 'pretraining')
    os.makedirs(EXPDIR, exist_ok=True)

    # ========== DATA ========================================
    train_loader = pretraining_pipeline(os.path.join(opt.data, 'train'),
                                        batch_size= 5 if opt.debug else opt.bs, 
                                        window_size=opt.window_size,
                                        shuffle=True,
                                        sampling=True,
                                        repeat=4,
                                        msk_frac=opt.probed,
                                        rnd_frac=opt.rs,
                                        same_frac=opt.rs)

    valid_loader = pretraining_pipeline(os.path.join(opt.data, 'val'),
                                        batch_size=5 if opt.debug else opt.bs,
                                        window_size=opt.window_size,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=opt.probed,
                                        rnd_frac=opt.rs,
                                        same_frac=opt.rs)

    # ======= MODEL ========================================
    model = get_ASTROMER(num_layers=opt.num_layers,
						 d_model=opt.head_dim*opt.num_heads,
						 num_heads=opt.num_heads,
						 dff=opt.mixer,
						 base=opt.pe_base,
						 rate=opt.dropout,
						 use_leak=False,
						 maxlen=opt.window_size)

    # ============================================================
    if opt.checkpoint != '-1':
        print('[INFO] Restoring previous training')
        model.load_weights(os.path.join(opt.checkpoint, 'weights'))
        
    if opt.scheduler:
        print('[INFO] Using Custom Scheduler')
        lr = CustomSchedule(d_model=int(opt.head_dim*opt.num_heads))
    else:
        lr = opt.lr

    model.compile(optimizer=Adam(lr, 
                             beta_1=0.9,
                             beta_2=0.98,
                             epsilon=1e-9,
                             name='astromer_optimizer'))

    with open(os.path.join(EXPDIR, 'config.toml'), 'w') as f:
        toml.dump(opt.__dict__, f)

    cbks = [TensorBoard(log_dir=os.path.join(EXPDIR, 'tensorboard')),
            EarlyStopping(monitor='val_loss', patience=opt.patience),
            ModelCheckpoint(filepath=os.path.join(EXPDIR, 'weights'),
                            save_weights_only=True,
                            save_best_only=True,
                            save_freq='epoch',
                            verbose=1)]

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
	parser.add_argument('--scheduler', action='store_true', help='Use Custom Scheduler during training')
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
