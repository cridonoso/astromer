import tensorflow as tf 
import wandb
import sys
import os

from src.models import get_ASTROMER_II
from src.data import load_data

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import AdamW
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks  import EarlyStopping
										 
WEIGHTS_FOLDER = './presentation/experiments/astromer_2/hp_results'
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
# =====================================================================================
# ===== SEARCH SPACE ==================================================================
# =====================================================================================
sweep_conf = {
	'name': 'ASTROMER_II',
	'method': 'bayes',
	'metric': {'goal': 'minimize', 'name': 'epoch/val_loss'},
	'early_terminate':{
	  'type': 'hyperband',
	  'min_iter': 20},
	'parameters': {
		'n_layers': {'values':[1, 2, 3]},
		'n_heads': {'values':[1, 2, 4]},
		'head_dim': {'values':[16, 32, 64]},
		'pedim': {'values':[64, 128, 256, 512]},
		'dff': {'values':[16, 32, 64, 128]},
		'dropout_rate': {'max': 0.5, 'min': 0.},
		'learning_rate':{'max': 1e-3, 'min': 1e-5},
		'probed':{'max':0.8 , 'min': 0.2},
		'optimizer':{'values':['adam', 'adamw']},
		'random_same':{'max':0.3 , 'min': 0.},
		'nsp_prob':{'value': 0.},
		'batch_size': {'value':1028},
		'window_size': {'value':200}
	}
}

def sweep_train(config=None):
	with wandb.init(config=config):
		config = wandb.config
		# =====================================================================================
		# ===== DATA ==========================================================================
		# =====================================================================================
		petrain_ds  = './data/records/macho_clean'
		# -------------------------------------------------------------------------------------
		# ========== DATA ========================================
		trainloader = load_data(dataset=os.path.join(petrain_ds, 'train'), 
								  batch_size=config.batch_size, 
								  probed=config.probed,
								  random_same=config.random_same,  
								  window_size=config.window_size, 
								  nsp_prob=config.nsp_prob, 
								  repeat=4, 
								  sampling=True,
								  off_nsp=True)
		validloader = load_data(dataset=os.path.join(petrain_ds, 'val'), 
								  batch_size=config.batch_size, 
								  probed=config.probed,  
								  random_same=config.random_same,
								  window_size=config.window_size, 
								  nsp_prob=config.nsp_prob, 
								  repeat=1, 
								  sampling=True,
								  off_nsp=True)

		# =====================================================================================
		# ===== MODEL =========================================================================
		# =====================================================================================

		astromer = get_ASTROMER_II(num_layers=config.n_layers,
								   num_heads=config.n_heads,
								   head_dim=config.head_dim,
								   mixer_size=config.dff,
								   dropout=config.dropout_rate,
								   pe_base=1000,
								   pe_dim=config.pedim,
								   pe_c=1,
								   window_size=config.window_size,
								   encoder_mode='normal',
								   average_layers=False,
								   off_nsp=True)

		lr = config.learning_rate
		if config.optimizer == 'adam':
			optimizer = Adam(lr)
		else:
			optimizer = AdamW(lr)

		astromer.compile(optimizer=optimizer)
		# =====================================================================================
		# ===== TRAINING ======================================================================
		# =====================================================================================
		N_EPOCHS = 10000
		astromer.fit(trainloader, 
					 epochs=N_EPOCHS, 
					 validation_data=validloader,
					 callbacks=[WandbModelCheckpoint(filepath=os.path.join(WEIGHTS_FOLDER, 'model'),
													 monitor='val_loss',
													 save_freq='epoch',
													 save_weights_only=True, 
													 save_best_only=True), 
								WandbMetricsLogger(log_freq='epoch'),
								EarlyStopping(monitor='val_loss',
											  patience = 20,
											  restore_best_weights=True)])

# =====================================================================================
# ===== WandB =========================================================================
# =====================================================================================
sweep_id = wandb.sweep(sweep_conf, project="hp-astromer-i-new")
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] # which gpu to use

if sys.argv[2] != '0':
	print('using previous id: ', sys.argv[2])
	sweep_id = sys.argv[2]

wandb.agent(sweep_id, function=sweep_train, count=100)