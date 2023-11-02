import tensorflow as tf
import argparse
import toml
import yaml
import sys
import os

from src.models.astromer_1 import get_ASTROMER, train_step, test_step

from src.training.utils import train
from src.data.zero import pretraining_pipeline
from datetime import datetime


def merge_metrics(**kwargs):
	merged = {}
	for key, value in kwargs.items():
		for subkey, subvalue in value.items():
			merged['{}_{}'.format(key, subkey)] = subvalue
	return merged


def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    ROOT = './presentation/experiments/astromer_1_pe/'
    #trial = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPDIR = os.path.join(ROOT, 'results', opt.pt_folder, '{}'.format(opt.exp_name))
    os.makedirs(EXPDIR, exist_ok=True)

    # ========== DATA ========================================
    train_loader = pretraining_pipeline(os.path.join(opt.data, 'train'),
                                        batch_size= 5 if opt.debug else opt.bs, 
                                        window_size=opt.window_size,
                                        shuffle=True,
                                        sampling=True,
                                        repeat=opt.repeat,
                                        msk_frac=opt.probed,
                                        rnd_frac=opt.rs,
                                        same_frac=opt.rs,
                                        key_format='1')

    valid_loader = pretraining_pipeline(os.path.join(opt.data, 'val'),
                                        batch_size=5 if opt.debug else opt.bs,
                                        window_size=opt.window_size,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=opt.probed,
                                        rnd_frac=opt.rs,
                                        same_frac=opt.rs,
                                        key_format='1')

    test_loader = pretraining_pipeline(os.path.join(opt.data, 'test'),
                                        batch_size=5 if opt.debug else opt.bs,
                                        window_size=opt.window_size,
                                        shuffle=False,
                                        sampling=True,
                                        msk_frac=opt.probed,
                                        rnd_frac=opt.rs,
                                        same_frac=opt.rs,
                                        key_format='1')

    # ======= MODEL ========================================
    file = open('{}/pe_config.yaml'.format(ROOT), "r")
    pe_config = yaml.load(file, Loader=yaml.FullLoader) 

    #path_pe_config = '{}/pe_config.toml'.format(ROOT)
    #with open(path_pe_config, mode="r") as fp:
    #    pe_config = toml.load(fp)

    astromer = get_ASTROMER(num_layers=opt.num_layers,
                            num_heads=opt.num_heads,
                            head_dim=opt.head_dim,
                            mixer_size=opt.mixer,
                            dropout=opt.dropout,
                            pe_type=opt.pe_type,
                            pe_config=pe_config,
                            pe_func_name=opt.pe_func_name,
                            residual_type=opt.residual_type,
                            window_size=opt.window_size,
                            encoder_mode=opt.encoder_mode,
                            average_layers=opt.avg_layers,
                            data_name=None)

    # ============================================================
    if opt.checkpoint != '-1':
        print('[INFO] Restoring previous training')
        astromer.load_weights(os.path.join(opt.checkpoint, 'weights', 'weights'))

    dict_pe = dict({opt.pe_func_name: pe_config[opt.pe_func_name]})
    file = open(os.path.join(EXPDIR, 'pe_config.yaml'), "w")
    yaml.dump(dict_pe, file)
    file.close()
    #with open(os.path.join(EXPDIR, 'pe_config.toml'), 'w') as f:
    #    toml.dump(dict_pe, f)

    print(astromer.summary())
    with open('{}/model_summary.txt'.format(EXPDIR), 'w') as f:
        astromer.summary(print_fn=lambda x: f.write(x + '\n'))

    astromer, \
	(best_train_metrics,
	best_val_metrics) = train(astromer,
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

    metrics = merge_metrics(train=best_train_metrics, 
                            val=best_val_metrics)

    with open(os.path.join(EXPDIR, 'train_val_metrics.toml'), 'w') as fp:
        toml.dump(metrics, fp)



if __name__ == '__main__':     
	parser = argparse.ArgumentParser()
    # General arguments
	parser.add_argument('--pt-folder', default='pretraining/P05R02', type=str,
					help='Project name')    
	parser.add_argument('--exp-name', default='macho_pe_nontrainable_repeat4', type=str,
					help='Project name')
	parser.add_argument('--data', default='./data/records/macho_clean', type=str,
					help='Data folder where tf.record files are located')
	parser.add_argument('--checkpoint', default='-1', type=str,
						help='Restore training by using checkpoints. This is the route to the checkpoint folder.')
	parser.add_argument('--gpu', default='0', type=str,
						help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')

    # Preprocessing
	parser.add_argument('--repeat', default=4, type=int,
                        help='Number of times for sampling windows from single LC')
	parser.add_argument('--window-size', default=200, type=int,
						help='windows size of the PSFs')

    # Arquitecture
	parser.add_argument('--encoder-mode', default='normal', type=str,
						help='normal - conditioned')
	parser.add_argument('--num-layers', default=2, type=int,
						help='Number of Attention Layers')
	parser.add_argument('--num-heads', default=4, type=int,
						help='Number of heads within the attention layer')
	parser.add_argument('--head-dim', default=64, type=int,
						help='Head dimension')
	parser.add_argument('--pe-type', default='APE', type=str,
						help='You can select: ["APE", "RPE", "MixPE"]')
	parser.add_argument('--pe-func-name', default='pe', type=str,
						help='You can select: ["not_pe_module", "use_t", "pe", "pe_mlp", "pe_rnn", "pe_tm", "pe_att"]')
	parser.add_argument('--residual-type', default=None, type=str,
						help='You can select: [None, "residual_in_all_attblocks", "residual_in_last_attblock"]')
	parser.add_argument('--mixer', default=128, type=int,
						help='Units to be used on the hidden layer of a feed-forward network that combines head outputs within an attention layer')
	parser.add_argument('--dropout', default=0.1, type=float,
						help='Dropout to use on the output of each attention layer (before mixer layer)')
	parser.add_argument('--avg-layers', action='store_true', help='If averaging outputs of the attention layers to form the final embedding. There is no avg if layers=1 ')

    # Hyperparameters
	parser.add_argument('--lr', default='scheduler', type=str,
						help='learning rate')
	parser.add_argument('--bs', default=2500, type=int,
						help='Batch size')
	parser.add_argument('--patience', default=20, type=int,
						help='Earlystopping threshold in number of epochs')
	parser.add_argument('--num-epochs', default=10000, type=int,
						help='Number of epochs')

    # Training type
	parser.add_argument('--probed', default=0.5, type=float,
						help='Probed percentage')
	parser.add_argument('--rs', default=0.2, type=float,
						help='Probed fraction to be randomized or unmasked')

	opt = parser.parse_args()        
	run(opt)
