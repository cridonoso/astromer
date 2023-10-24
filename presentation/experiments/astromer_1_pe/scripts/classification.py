import tensorflow as tf
import pandas as pd
import argparse
import toml
import os

from presentation.experiments.classifiers import train_classifier, build_mlp_att
from src.models.astromer_1 import get_ASTROMER, build_input
from src.data import load_data


def set_trainable_layers(model, clf_science_case, layers_config):
	print(f'Processing science_case: {clf_science_case}...')

	for layer in model.get_layer('encoder').layers:
		print(f'Name layers: {layer.name}')		
		if layer.name == 'inp_transform':  
			layer.trainable = layers_config[clf_science_case]['ff1_layer']
		elif layer.name.find('att_') != -1:
			layer.trainable = layers_config[clf_science_case]['att_layer']

	for layer in model.layers:
		if layer.name != 'encoder':
			layer.trainable = layers_config[clf_science_case]['ff2_layer']

	return model


def run(opt):
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

	ROOT = './presentation/experiments/astromer_1_pe/'

	# =========================================================================
	# =============== PATH WEIGHTS SELECTION ==================================
	# =========================================================================

	if opt.last_exp.find('finetuning'):
		path_folder = os.path.join(opt.last_exp, 
								   opt.subdataset,
								   'fold_'+str(opt.fold), 
								   '{}_{}'.format(opt.subdataset, opt.spc)) 
		
		print('[INFO] You going to load the finetuning weights')
	
	else:
		path_folder = opt.last_exp
		print('[INFO] You going to load the pretraining weights')

	# =========================================================================
	# =============== LOADING SETTING =========================================
	# =========================================================================

	# General config for frezee and trainable layers
	with open('./{}/layers_config.toml'.format(ROOT), mode="r") as f:
		layers_config = toml.load(f)

	# General config
	with open(os.path.join(ROOT, path_folder, 'config.toml'), 'r') as f:
		model_config = toml.load(f)

	with open(os.path.join(ROOT, path_folder, 'pe_config.toml'), 'r') as f:
		pe_config = toml.load(f)
	
	# Update model config
	dict_model_config = model_config.copy()
	dict_model_config['Classification'] = opt.__dict__

	# Update PE config
	dict_pe = dict({model_config['Pretraining']['pe_type']: pe_config[model_config['Pretraining']['pe_type']]})
	dict_pe[model_config['Pretraining']['pe_type']]['pe_trainable'] = layers_config[opt.clf_science_case]['pe_layer']

	# ====================================================================================
	# =============== DATA LOAD  =========================================================
	# ====================================================================================
	CLASSIFICATION_DATA = os.path.join('./data/records', 
									   opt.subdataset,
									   'fold_'+str(opt.fold), 
									   '{}_{}'.format(opt.subdataset, opt.spc)) 

	num_cls = pd.read_csv(os.path.join(CLASSIFICATION_DATA, 'objects.csv')).shape[0]

	train_loader = load_data(dataset=os.path.join(CLASSIFICATION_DATA, 'train'), 
							 batch_size= 5 if opt.debug else opt.bs, 
							 probed=1.,
							 random_same=0.,  
							 window_size=model_config['Pretraining']['window_size'], 
							 off_nsp=True,
							 nsp_prob=0., 
							 repeat=1, 
							 sampling=False,
							 shuffle=True,
							 num_cls=num_cls)
	valid_loader = load_data(dataset=os.path.join(CLASSIFICATION_DATA, 'val'), 
							 batch_size= 5 if opt.debug else opt.bs, 
							 probed=1.,
							 random_same=0.,  
							 window_size=model_config['Pretraining']['window_size'], 
							 off_nsp=True,
							 nsp_prob=0., 
							 repeat=1, 
							 sampling=False,
							 num_cls=num_cls)
	test_loader = load_data(dataset=os.path.join(CLASSIFICATION_DATA, 'test'), 
							 batch_size= 5 if opt.debug else opt.bs, 
							 probed=1.,
							 random_same=0.,  
							 window_size=model_config['Pretraining']['window_size'], 
							 off_nsp=True, 
							 nsp_prob=0., 
							 repeat=1, 
							 sampling=False,
							 num_cls=num_cls)

	if opt.debug:
		train_loader = train_loader.take(1)
		valid_loader = valid_loader.take(1)
		test_loader  = test_loader.take(1)

	# ====================================================================================
	# =============== LOADING PRETRAINED MODEL ===========================================
	# ====================================================================================

	astromer = get_ASTROMER(num_layers=model_config['Pretraining']['num_layers'],
							num_heads=model_config['Pretraining']['num_heads'],
							head_dim=model_config['Pretraining']['head_dim'],
							mixer_size=model_config['Pretraining']['mixer'],
							dropout=model_config['Pretraining']['dropout'],
							pe_type=model_config['Pretraining']['pe_type'],
							pe_config=dict_pe,
							window_size=model_config['Pretraining']['window_size'],
							encoder_mode=model_config['Pretraining']['encoder_mode'],
							average_layers=model_config['Pretraining']['avg_layers'])

	astromer.load_weights(os.path.join(ROOT, path_folder, 'weights', 'weights'))
	print('[INFO] Weights loaded')

	# ====================================================================================
	# =============== CLASSIFICATION TASK  ===============================================
	# ====================================================================================
	
	CLFWEIGHTS = os.path.join(ROOT,
						   	  opt.clf_folder,
							  opt.subdataset, 
							  'fold_'+str(opt.fold), 
							  opt.subdataset+'_'+str(opt.spc))

	inp_placeholder = build_input(model_config['Pretraining']['window_size'])
	encoder = astromer.get_layer('encoder')
	embedding = encoder(inp_placeholder)
	embedding = embedding*(1.-inp_placeholder['att_mask'])
	embedding = tf.math.divide_no_nan(tf.reduce_sum(embedding, axis=1), 
									  tf.reduce_sum(1.-inp_placeholder['att_mask'], axis=1))
	
	model = build_mlp_att(embedding, inp_placeholder, num_cls, opt.clf_name)
	model = set_trainable_layers(model, opt.clf_science_case, layers_config)
	model.compile(optimizer=tf.keras.optimizers.Adam(opt.lr),
				  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
				  metrics=['accuracy'])

	print(model.summary())
	with open('{}/model_summary.txt'.format(CLFWEIGHTS), 'w') as f:
		model.summary(print_fn=lambda x: f.write(x + '\n'))

	# Save PE configuration
	os.makedirs(CLFWEIGHTS, exist_ok=True)
	with open(os.path.join(CLFWEIGHTS, 'pe_config.toml'), 'w') as f:
		toml.dump(dict_pe, f)

	summary_clf = train_classifier(model,
								   train_loader=train_loader,
								   valid_loader=valid_loader, 
								   test_loader=test_loader,
								   num_epochs=opt.num_epochs,
								   patience=opt.patience,
								   project_path=CLFWEIGHTS,
								   clf_name=opt.clf_name,
								   debug=opt.debug,
								   argparse_dict=dict_model_config)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='1', type=str, help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--subdataset', default='alcock', type=str, help='Data folder where tf.record files are located')
	parser.add_argument('--fold', default=0, type=int, help='Fold to use')
	parser.add_argument('--spc', default=500, type=int, help='Samples per class')

	parser.add_argument('--last-exp', default='results/finetuning/P02R01/macho_pe_nontrainable/FF1_PE_ATT_FF2', 
					 type=str, help='pretrained model folder')

	parser.add_argument('--clf-folder', default='results/classification/P02R01/macho_pe_nontrainable/FF1_PE_ATT_FF2', 
					 type=str, help='pretrained model folder')
	parser.add_argument('--clf-science-case', default='FF1_PE_ATT_FF2', type=str, help='Layers trainables')
	parser.add_argument('--debug', action='store_true', help='a debugging flag to be used when testing.')

	parser.add_argument('--lr', default='1e-3', type=str, help='learning rate')
	parser.add_argument('--bs', default=512, type=int,	help='Batch size')
	parser.add_argument('--patience', default=20, type=int,	help='Earlystopping threshold in number of epochs')
	parser.add_argument('--num-epochs', default=10000, type=int, help='Number of epochs')
	parser.add_argument('--clf-name', default='att_mlp', type=str, help='classifier name')

	opt = parser.parse_args()        
	run(opt)
