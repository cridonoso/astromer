import tensorflow as tf 
import pandas as pd
import wandb
import sys
import toml
import os

from tensorflow.keras.callbacks  import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbMetricsLogger

from presentation.experiments.astromer_2.classification import create_classifier
from src.models import get_ASTROMER_II
from src.data.loaders import load_light_curves
from src.data import load_data

from sklearn.metrics import precision_recall_fscore_support

# g34htgii 
DEBUG = False
ROOT = './presentation/experiments/astromer_2/results'
MASTER_PROJECT_NAME = 'downstream_a2_50probed'
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

# =====================================================================================
# ===== SEARCH SPACE ==================================================================
# =====================================================================================
sweep_conf = {
	'name': 'ASTROMER_II',
	'method': 'grid',
	'metric': {'goal': 'maximize', 'name': 'val_acc'},
	'parameters': {
		'pt_model': {'values':['nsp_cond/1_4_64_rmse_0.5', 'nsp_cond/2_4_64_rmse_0.5',
							   'nsp_normal/1_4_64_rmse_0.5', 'nsp_normal/1_4_64_rmse_0.5', 
                               'nsp_concat/1_4_64_rmse_0.5', 'nsp_concat/1_4_64_rmse_0.5']},
		'fold':{'values':[0, 1, 2]},
		'subdataset':{'values':['atlas', 'alcock']},
		'clf_name':{'values':['mlp_att', 'mlp_cls', 'mlp_all']},
	}
}

def check_if_exist_finetuned_weights(config, project_name):
	api = wandb.Api()
	runs = api.runs(project_name)
	for run in runs:
		if run.config['subdataset'] == config.subdataset and \
		   run.config['fold']== config.fold and \
		   run.state == 'finished':
			return True
	return False
	
def sweep_train(config=None):
	with wandb.init(config=config):
		config = wandb.config                   
	   
		# ====================================================================================
		# PRE-TRAINING =======================================================================
		# ====================================================================================
		PTWEIGTHS = os.path.join(ROOT, config.pt_model, 'pretraining')


		with open(os.path.join(PTWEIGTHS, 'config.toml'), 'r') as f:
			model_config = toml.load(f)
			wandb.log(model_config)

		astromer = get_ASTROMER_II(num_layers=model_config['layers'],
								   num_heads=model_config['nh'],
								   head_dim=model_config['hdim'],
								   mixer_size=model_config['mixer'],
								   dropout=model_config['dropout'],
								   pe_base=1000,
								   pe_dim=128,
								   pe_c=1,
								   window_size=model_config['ws'],
								   encoder_mode=model_config['encoder_mode'])

		print('[INFO] LOADING PRETRAINED WEIGHTS')
		astromer.load_weights(os.path.join(PTWEIGTHS, 'weights')).expect_partial()

		# =====================================================================================
		# === FINETUNING STEP =================================================================
		# =====================================================================================  
		DOWNSTREAM_DATA = os.path.join('./data/records', 
									   config.subdataset,
									   'fold_'+str(config.fold), 
									   config.subdataset+'_20')
		FTWEIGTHS = os.path.join(ROOT, 
								 config.pt_model,
								 'finetuning',                                     
								 config.subdataset,
								 'fold_'+str(config.fold), 
								 config.subdataset+'_20')     
		

		does_it_exists = check_if_exist_finetuned_weights(config, MASTER_PROJECT_NAME)

		if os.path.isfile(os.path.join(FTWEIGTHS, 'checkpoint')) and does_it_exists:
			print('[INFO] FINETUNED MODEL FOUND. LOADING WEIGHTS')
			astromer.load_weights(os.path.join(FTWEIGTHS, 'weights')).expect_partial()
		else:
			print('[INFO] FINETUNING FROM SCRATCH')

			train_batches = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'train'), 
									  batch_size=32 if DEBUG else model_config['bs'], 
									  probed=model_config['probed'],  
									  window_size=model_config['ws'], 
									  nsp_prob=model_config['nsp_prob'], 
									  repeat=1, 
									  sampling=False)
			valid_batches = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'val'), 
									  batch_size=32 if DEBUG else model_config['bs'], 
									  probed=model_config['probed'],  
									  window_size=model_config['ws'], 
									  nsp_prob=model_config['nsp_prob'], 
									  repeat=1, 
									  sampling=False)
			test_batches = load_data(dataset=os.path.join(DOWNSTREAM_DATA, 'test'), 
									  batch_size=32 if DEBUG else model_config['bs'], 
									  probed=model_config['probed'],  
									  window_size=model_config['ws'], 
									  nsp_prob=model_config['nsp_prob'], 
									  repeat=1, 
									  sampling=False)

			optimizer = AdamW(model_config['lr'], beta_1=0.9, beta_2=0.98, epsilon=1e-9)
			astromer.compile(rmse_factor=model_config['rmse_factor'], 
							 bce_factor=1.-model_config['rmse_factor'], optimizer=optimizer)
			
			cbks =  [
				ModelCheckpoint(
					filepath=os.path.join(FTWEIGTHS, 'weights'),
					save_weights_only=True,
					monitor='val_loss',
					save_best_only=True),
				EarlyStopping(monitor='val_loss',
					patience = 20,
					restore_best_weights=True),
				TensorBoard(
					log_dir = os.path.join(FTWEIGTHS, 'logs'),
					histogram_freq=1,
					write_graph=True),
				WandbMetricsLogger(log_freq='epoch')]

			astromer.fit(train_batches, 
						 epochs=2 if DEBUG else model_config['epochs'], 
						 validation_data=valid_batches,
						 callbacks=cbks) 

			print('[INFO] Testing')
			acc, bce, loss, r2, rmse = astromer.evaluate(test_batches)   
			wandb.log({'ft_test_acc': acc, 'ft_test_r2':r2, 'ft_test_rmse':rmse, 'ft_test_bce':bce})

			with open(os.path.join(FTWEIGTHS, 'config.toml'), 'w') as f:
				toml.dump(model_config, f)
		# ============================================================================
		# =========== CLASSIFICATION==================================================
		# ============================================================================
		CLFWEIGHTS = os.path.join(ROOT, 
								  config.pt_model,
								  'classification', 
								  config.subdataset, 
								  'fold_'+str(config.fold), 
								  config.subdataset+'_20')
		os.makedirs(CLFWEIGHTS, exist_ok=True)

		num_cls = pd.read_csv(
				os.path.join(DOWNSTREAM_DATA, 'objects.csv')).shape[0]
		
		train_batches = load_light_curves(os.path.join(DOWNSTREAM_DATA, 'train'), 
										  num_cls=num_cls,
										  batch_size=32 if DEBUG else 512, 
										  window_size=model_config['ws'], 
										  repeat=1,
										  cache=True, 
										  njobs=None)
		valid_batches = load_light_curves(os.path.join(DOWNSTREAM_DATA, 'val'), 
										  num_cls=num_cls,
										  batch_size=32 if DEBUG else 512, 
										  window_size=model_config['ws'], 
										  repeat=1,
										  cache=True, 
										  njobs=None)
		test_batches = load_light_curves(os.path.join(DOWNSTREAM_DATA, 'test'), 
										  num_cls=num_cls,
										  batch_size=32 if DEBUG else 512, 
										  window_size=model_config['ws'], 
										  repeat=1,
										  cache=True, 
										  njobs=None)

		clf_model = create_classifier(FTWEIGTHS, model_config['ws'], num_cls, clf_name=config.clf_name)

		clf_model.compile(optimizer=Adam(1e-3),
						  loss=CategoricalCrossentropy(from_logits=True))
		cbks =  [
			ModelCheckpoint(
				filepath=os.path.join(CLFWEIGHTS, 'weights'),
				save_weights_only=True,
				monitor='val_loss',
				save_best_only=True),
			EarlyStopping(monitor='val_loss',
				patience = 20,
				restore_best_weights=True),
			TensorBoard(
				log_dir = os.path.join(CLFWEIGHTS, 'logs'),
				histogram_freq=1,
				write_graph=True),
			WandbMetricsLogger(log_freq='epoch')]

		history = clf_model.fit(train_batches,
								epochs= 2 if DEBUG else model_config['epochs'],
								callbacks=cbks,
								validation_data=valid_batches)

		y_pred = clf_model.predict(test_batches)
		y_true = tf.concat([y['label'] for _, y in test_batches], 0)
		pred_labels = tf.argmax(y_pred, 1)
		true_labels = tf.argmax(y_true, 1)
		p, r, f, _ = precision_recall_fscore_support(true_labels,
													 pred_labels,
													 average='macro',
													 zero_division=0.)
	
		wandb.log({'clf_test_precision': p, 'clf_test_recall': r, 'clf_test_f1': f})


# =====================================================================================
# ===== WandB =========================================================================
# =====================================================================================
sweep_id = wandb.sweep(sweep_conf, project=MASTER_PROJECT_NAME)

if sys.argv[2] != '0':
	print('using previous id: ', sys.argv[2])
	sweep_id = sys.argv[2]

wandb.agent(sweep_id, function=sweep_train, count=100)

