import tensorflow as tf 
import wandb
import sys
import json
import pandas as pd
import os

from .utils import load_pt_data, load_clf_data, create_classifier, classify, get_callbacks, check_if_exist_finetuned_weights
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.optimizers import Adam

from src.models import get_ASTROMER_II

# g34htgii 
DEBUG = False
MASTER_PROJECT_NAME = 'nsp_adamw'
ROOT = './presentation/experiments/astromer_2/results'
EXPDIR = os.path.join(ROOT, 'results', MASTER_PROJECT_NAME)
os.makedirs(EXPDIR, exist_ok=True)

# =====================================================================================
# ===== SEARCH SPACE ==================================================================
# =====================================================================================
sweep_conf = {
    'name': 'ASTROMER_II',
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'epoch/val_accuracy'},
    'parameters': {
        'pt_data':{'value':'./data/records/macho_clean'},
        'n_layers': {'values':[3]},
        'fold':{'values':[0, 1, 2]},
        'subdataset':{'values':['atlas', 'alcock']},
        'clf_name':{'values':['mlp_att', 'mlp_cls', 'mlp_att_lite']},
        'n_heads': {'value':6},
        'head_dim': {'value':64},
        'dff': {'value':256},
        'dropout_rate': {'value': 0.1},
        'learning_rate':{'value':1e-5},
        'window_size': {'value':200},
        'probed':{'value': 0.6},
        'rand': {'value':0.2},
        'nsp_prob':{'value':0.5},
        'nsp_fraction':{'value':0.5},
        'n_epochs': {'value':10000},
        'batch_size':{'value':2500}
    }
}


def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config                   
       
        # ====================================================================================
        # PRE-TRAINING =======================================================================
        # ====================================================================================
        model_name = '{}_{}_{}'.format(config.n_layers, config.n_heads, config.head_dim)
        PTWEIGTHS = os.path.join(EXPDIR, model_name, 'pretraining')
        
        d_model = config.head_dim*config.n_heads
        astromer =  get_ASTROMER_II(num_layers=config.n_layers,
                                    d_model=d_model,
                                    num_heads=config.n_heads,
                                    dff=config.dff,
                                    base=10000,
                                    dropout=config.dropout_rate,
                                    maxlen=config.window_size,
                                    pe_c=2)          
        
        if os.path.exists(os.path.join(PTWEIGTHS, 'results.json')):
            print('[INFO] LOADING PRETRAINED WEIGHTS')
            astromer.load_weights(os.path.join(PTWEIGTHS, 'weights'))
        else:
            print('[INFO] TRAINING FROM SCRATCH')
            optimizer = AdamW(config.learning_rate)
            astromer.compile(optimizer=optimizer)
            loader = load_pt_data(config, sampling=True, debug=DEBUG)
            cbks = get_callbacks(PTWEIGTHS, monitor='val_loss')
            astromer.fit(loader['train'], 
                     epochs=2 if DEBUG else config.n_epochs, 
                     validation_data=loader['val'],
                     callbacks=cbks)      
            
            acc, bce, loss, r2, rmse = astromer.evaluate(loader['test'])   
            with open(os.path.join(PTWEIGTHS, 'results.json'), 'w') as fp:
                json.dump({'test_acc': acc, 
                           'test_r2':r2, 
                           'test_rmse':rmse, 
                           'test_bce':bce}, fp)

        # =====================================================================================
        # === FINETUNING STEP =================================================================
        # =====================================================================================  
        DOWNSTREAM_DATA = os.path.join('./data/records', 
                                       config.subdataset,
                                       'fold_'+str(config.fold), 
                                       config.subdataset+'_20')
        FTWEIGTHS = os.path.join(EXPDIR, 
                                 model_name,
                                 'finetuning',                                     
                                 config.subdataset,
                                 'fold_'+str(config.fold), 
                                 config.subdataset+'_20')     
        
        does_it_exists = check_if_exist_finetuned_weights(config, MASTER_PROJECT_NAME)

        if os.path.isfile(os.path.join(FTWEIGTHS, 'checkpoint')) and does_it_exists:
            print('[INFO] FINETUNED MODEL FOUND. LOADING WEIGHTS')
            astromer.load_weights(os.path.join(FTWEIGTHS, 'weights'))
        else:
            print('[INFO] FINETUNING FROM SCRATCH')
            optimizer = Adam(config.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
            astromer.compile(optimizer=optimizer)

            loader = load_pt_data(config, sampling=False, datapath=DOWNSTREAM_DATA, debug=DEBUG)
            
            cbks = get_callbacks(FTWEIGTHS, monitor='val_loss')
            astromer.fit(loader['train'], 
                         epochs=2 if DEBUG else config.n_epochs, 
                         validation_data=loader['val'],
                         callbacks=cbks) 
            acc, bce, loss, r2, rmse = astromer.evaluate(loader['test'])   
            wandb.log({'test_acc': acc, 'test_r2':r2, 'test_rmse':rmse, 'test_bce':bce})

        # ============================================================================
        # =========== CLASSIFICATION==================================================
        # ============================================================================
        CLFWEIGHTS = os.path.join(EXPDIR, 
                                  model_name,
                                  'classification', 
                                  config.subdataset, 
                                  'fold_'+str(config.fold), 
                                  config.subdataset+'_20')
        
        num_cls = pd.read_csv(
                os.path.join(DOWNSTREAM_DATA, 'objects.csv')).shape[0]
        
        data = load_clf_data(config, 
                             batch_size=512, 
                             num_cls=num_cls, 
                             datapath=DOWNSTREAM_DATA, 
                             debug=DEBUG)

        clf_model = create_classifier(astromer, 
                                      config, 
                                      num_cls=num_cls, 
                                      train_astromer=False, 
                                      name=config.clf_name)

        clf_model, backlog_df = classify(clf_model, 
                                         data, 
                                         CLFWEIGHTS, 
                                         lr=1e-3, 
                                         model_name=clf_model.name,
                                         debug=DEBUG)
        wandb.log(backlog_df)


# =====================================================================================
# ===== WandB =========================================================================
# =====================================================================================
sweep_id = wandb.sweep(sweep_conf, project=MASTER_PROJECT_NAME)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

if sys.argv[2] != '0':
    print('using previous id: ', sys.argv[2])
    sweep_id = sys.argv[2]

wandb.agent(sweep_id, function=sweep_train, count=100)

