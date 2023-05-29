import tensorflow as tf 
import wandb
import sys
import os

from src.models import get_ASTROMER, build_input
from src.data import pretraining_pipeline

from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks  import EarlyStopping
                                         
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1] # which gpu to use

WEIGHTS_FOLDER = './presentation/scripts/hp_results'
os.makedirs(WEIGHTS_FOLDER, exist_ok=True)
# =====================================================================================
# ===== SEARCH SPACE ==================================================================
# =====================================================================================
sweep_conf = {
    'name': 'ASTROMER_I',
    'method': 'bayes',
    'metric': {'goal': 'minimize', 'name': 'epoch/val_loss'},
    'early_terminate':{
      'type': 'hyperband',
      'min_iter': 5},
    'parameters': {
        'n_layers': {'values':[1, 2]},
        'n_heads': {'values':[1, 2, 4]},
        'head_dim': {'values':[16, 32, 64]},
        'dff': {'values':[16, 32, 64, 128]},
        'dropout_rate': {'max': 0.5, 'min': 0.},
        'learning_rate':{'max': 1e-1, 'min': 1e-5},
    	'batch_size': {'value':2500},
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
		trainloader = pretraining_pipeline(os.path.join(petrain_ds, 'train'), 
										   config.batch_size, 
                                           config.window_size, 
                                           .5, .2, .2,
		                                   sampling=True, 
                                           shuffle=True, 
                                           repeat=4, 
                                           num_cls=None,
		                                   normalize="zero-mean", 
                                           cache=True)
		validloader = pretraining_pipeline(os.path.join(petrain_ds, 'val'), 
										   config.batch_size, 
                                           config.window_size, 
                                           .5, .2, .2,
		                                   sampling=True, 
                                           shuffle=False, 
                                           repeat=1, 
                                           num_cls=None,
		                                   normalize="zero-mean", 
                                           cache=True)
		# trainloader = trainloader.take(1)
		# validloader = validloader.take(1)
		# =====================================================================================
		# ===== MODEL =========================================================================
		# =====================================================================================
		d_model      = config.head_dim*config.n_heads
		astromer 	 =  get_ASTROMER(num_layers=config.n_layers,
									 d_model=d_model,
									 num_heads=config.n_heads,
									 dff=config.dff,
									 base=1000,
									 dropout=config.dropout_rate,
									 maxlen=config.window_size,
									 pe_c=1.)
		optimizer = Adam(config.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
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
sweep_id = wandb.sweep(sweep_conf, project="hp-astromer-zero")
wandb.agent(sweep_id, function=sweep_train, count=50)
