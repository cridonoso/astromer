import tomli, tomli_w
import pandas as pd
import itertools
import os

from datetime import datetime

# ===================================================================================================
# OPENING TEMPLATE CONFIG FILE and BUILDER ==========================================================
# Template should contain all the parameters needed for running the pipeline
# Builder contains all the posible values for hyperparameters
# ===================================================================================================
template_path = './presentation/experiments/astromer_0/template.toml'
builder_path  = './presentation/experiments/astromer_0/config_builder.toml'
with open(template_path, mode="rb") as fp:
    config = tomli.load(fp)

with open(builder_path, mode="rb") as fp:
    builder = tomli.load(fp)

exp_folder    = os.path.join(builder['experiment']['exp_folder'], builder['experiment']['exp_name'])
config_folder = os.path.join(exp_folder, 'config_files')
os.makedirs(exp_folder, exist_ok=True)
os.makedirs(config_folder, exist_ok=True)

# ====================================================================================================
# === LIST EXPERIMENTS ===============================================================================
# ====================================================================================================
hparam_values = {}
hparam_names  = {}
for key, value in builder.items():
    if key == 'experiment': continue
    hparam_values[key] = list(itertools.product(*[subval for subkey, subval in value.items()]))
    hparam_names[key] = [subkey for subkey, _ in value.items()]

t = list(itertools.product(*[value for key, value in hparam_values.items()]))
combinations = [tuple(itertools.chain(*tt)) for tt in t]
params_names = [ttt for _, tt in hparam_names.items() for ttt in tt]

# ====================================================================================================
# === SUMMARY RESULTS TABLE ==========================================================================
# ====================================================================================================
summary_df = pd.DataFrame(columns=['eid',
                                   'date',
                                   'step',
                                   'target', 
                                   'fold', 
                                   'spc',
                                   'rmse', 
                                   'r_square',
                                   'val_rmse',
                                   'val_r_square',
                                   'test_rmse',
                                   'test_r_square',
                                   'model',
                                   'acc',
                                   'loss',
                                   'val_acc',
                                   'val_loss',
                                   'test_precision',
                                   'test_recall',
                                   'test_f1',
                                   'elapsed'
                                   ])
summary_df.to_csv(os.path.join(exp_folder, 'results.csv'), index=False)

# ====================================================================================================
# === ITERATE OVER THE EXPERIMENTS ===================================================================
# ====================================================================================================
for eid, param in enumerate(combinations):
    param_dict = dict(zip(params_names, param))
    
    config['astromer']['layers']        = param_dict['num_layers']
    config['astromer']['heads']         = param_dict['num_heads']
    config['astromer']['head_dim']      = param_dict['head_dim']
    config['astromer']['dff']           = param_dict['dff']
    config['astromer']['dropout']       = param_dict['dropout']
    config['astromer']['window_size']   = param_dict['windows_size']


    config['general']['creation_date'] = datetime.today().strftime('%m-%d-%Y')

    config['masking']['mask_frac'] = param_dict['probed_ptge']
    config['masking']['rnd_frac']  = param_dict['rndsame_ptge']
    config['masking']['same_frac'] = param_dict['rndsame_ptge']

    config['positional']['alpha']  = param_dict['pe_alpha']

    config['pretraining']['lr'] = param_dict['learn_rate']
    config['pretraining']['scheduler']= False
    config['pretraining']['data']['batch_size'] = param_dict['batch_size']
    config['pretraining']['data']['path'] = param_dict['dataset']
    config['pretraining']['data']['target'] = ''
    config['pretraining']['data']['fold'] = 0
    config['pretraining']['data']['spc'] = ''    
    
    name_pt = '{}_{}_{}.{}'.format(config['astromer']['layers'], 
                                   config['astromer']['window_size'],
                                   config['masking']['mask_frac'],
                                   config['masking']['rnd_frac'])
    
    config['pretraining']['exp_path'] = os.path.join(exp_folder, 'pretraining', name_pt)
    
    config['classification']['train_astromer'] = False
    config['classification']['sci_case'] = 'a'

    config['finetuning']['data']['path']   = param_dict['dataset_2']
    config['finetuning']['data']['target'] = param_dict['dataset_2'].split('/')[3]
    config['finetuning']['data']['fold']   = int(param_dict['dataset_2'].split('/')[4].split('_')[-1])
    config['finetuning']['data']['spc']    = int(param_dict['dataset_2'].split('/')[-1].split('_')[-1])
    config['finetuning']['batch_size']     = param_dict['batch_size']

    config['classification']['data']['path']   = config['finetuning']['data']['path']
    config['classification']['data']['target'] = config['finetuning']['data']['target'] 
    config['classification']['data']['fold']   = config['finetuning']['data']['fold']
    config['classification']['data']['spc']    = config['finetuning']['data']['spc']
    config['classification']['batch_size']     = param_dict['batch_size_2']


    config['finetuning']['exp_path']     = os.path.join(exp_folder,
                                                        config['finetuning']['data']['target'],
                                                       'finetuning')

    config['classification']['exp_path'] = os.path.join(exp_folder,
                                                        config['classification']['data']['target'],
                                                       'classification')

    config['finetuning']['weights']     = config['pretraining']['exp_path']
    config['classification']['weights'] = config['finetuning']['exp_path']

    # SAVING CONFIG FILE
    with open(os.path.join(config_folder, f'exp_{eid}.toml'), mode="wb") as fp:
        tomli_w.dump(config, fp)



