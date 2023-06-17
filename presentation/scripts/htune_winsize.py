import tensorflow as tf 
import wandb
import sys
import os

from src.models import get_ASTROMER, build_input
from src.data import pretraining_pipeline
from src.training import CustomSchedule

from tensorflow.keras.optimizers import Adam
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.callbacks  import EarlyStopping, ModelCheckpoint
                                         

project_name = 'winsize'
WEIGHTS_FOLDER = './presentation/scripts/hp_results'
os.makedirs(os.path.join(WEIGHTS_FOLDER, project_name), exist_ok=True)

# =====================================================================================
# ===== SEARCH SPACE ==================================================================
# =====================================================================================
sweep_conf = {
    'name': 'ASTROMER_I',
    'method': 'grid',
    'metric': {'goal': 'minimize', 'name': 'epoch/val_loss'},
    'parameters': {
        'n_layers': {'values':[1],},
        'n_heads': {'value':4},
        'head_dim': {'value':64},
        'dff': {'value':64},
        'dropout_rate': {'value': 0.3955},
        'learning_rate':{'value':1e-5},
        'window_size': {'values':[20, 50, 100, 200, 500, 800]},
        'probed': {'values':[0.5]},
        'rand': {'value':0.2}
    }
}

def get_batch_size(model, bytes_per_param=4, window_size=None):
    params = model.count_params()    
    if window_size > 200:
        bs = int(300*595841/params)
    else:
        bs = int(3000*595841/params)
    return min(bs, 3000)

def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        print(config)
                
        SAVEPATH = os.path.join(WEIGHTS_FOLDER, project_name, ':.0f'.format(config.window_size))
        
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
                                     pe_c=2.,
                                     no_train=False)

        batch_size = get_batch_size(astromer, window_size=config.window_size)
        print('BATCH SIZE: ', batch_size)
        # =====================================================================================
        # ===== DATA ==========================================================================
        # =====================================================================================
        petrain_ds  = './data/records/macho_clean'
        # -------------------------------------------------------------------------------------
        trainloader = pretraining_pipeline(os.path.join(petrain_ds, 'train'), 
                                           batch_size, 
                                           config.window_size, 
                                           config.probed, config.rand, config.rand,
                                           sampling=True, 
                                           shuffle=True, 
                                           repeat=4, 
                                           num_cls=None,
                                           normalize="zero-mean", 
                                           cache=True)
        validloader = pretraining_pipeline(os.path.join(petrain_ds, 'val'), 
                                           batch_size, 
                                           config.window_size, 
                                           config.probed, config.rand, config.rand,
                                           sampling=True, 
                                           shuffle=False, 
                                           repeat=1, 
                                           num_cls=None,
                                           normalize="zero-mean", 
                                           cache=True)
        
        # =====================================================================================
        # ===== TRAINING ======================================================================
        # =====================================================================================
        lr = 1e-5 #CustomSchedule(d_model)
        optimizer = Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        astromer.compile(optimizer=optimizer)
        
        N_EPOCHS = 10000
        astromer.fit(trainloader, 
                     epochs=N_EPOCHS, 
                     validation_data=validloader,
                     callbacks=[WandbMetricsLogger(log_freq='epoch'),
                                EarlyStopping(monitor='val_loss',
                                              patience = 20,
                                              restore_best_weights=True),
                                ModelCheckpoint(filepath=os.path.join(SAVEPATH, 'weights'),
                                                save_weights_only=True,
                                                monitor='val_loss',
                                                save_best_only=True)])

# =====================================================================================
# ===== WandB =========================================================================
# =====================================================================================
sweep_id = wandb.sweep(sweep_conf, project=project_name)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

#7beif1u9
if sys.argv[2] != '0':
    print('using previous id: ', sys.argv[2])
    sweep_id = sys.argv[2]

wandb.agent(sweep_id, function=sweep_train, count=1)

