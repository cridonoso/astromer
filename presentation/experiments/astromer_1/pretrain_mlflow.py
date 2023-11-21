import tensorflow as tf
import argparse
import sys
import mlflow as mf
import optuna
import yaml
import os

from src.models.astromer_1 import get_ASTROMER, train_step, test_step

from src.training.utils import train
from datetime import datetime

from src.data.zero import pretraining_pipeline
from src.utils import (mf_check_run_exists,
                       mf_create_or_get_experiment_id,
                       mf_set_experiment,
                       mf_set_tracking_uri)

OPTUNA_CONFIG = "astromer_optuna.yaml"
def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    ROOT = './presentation/experiments/astromer_1/'
    trial = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXPDIR = os.path.join(ROOT, 'results', opt.exp_name, trial, 'pretraining')
    os.makedirs(EXPDIR, exist_ok=True)
    
    with open(os.path.join(opt.optuna_config, OPTUNA_CONFIG), 'r') as f:
        optuna_config = yaml.safe_load(f)
        
    category_mapping = {
        "int": optuna.distributions.IntDistribution,
        "float": optuna.distributions.FloatDistribution,
        "category": optuna.distributions.CategoricalDistribution
    }
    
    def generate_distritbutions(config, mapping:dict)->dict:
        
        distributions = {}
        
        
        for field in config:
            #Validate Field Type
            field_type = config[field]["type"]
            dist = mapping.get(field_type, None)
            if dist is None:
                types = mapping.keys()
                raise ValueError("""Type {field_type} is not present in the mapping, please update the mapping or provide
                                 one of the following {types}""".format(field_type=field_type, types=types))
            
            #Create Distribution
            if field_type == "category":
                choices = config[field]["values"]
                distributions[field] = mapping[field_type](choices=choices)
            else:
                low = config[field]["low"]
                high = config[field]["high"]
                distributions[field] = mapping[field_type](low=low, high=high)
        
        return distributions
    
    distributions = generate_distritbutions(config=optuna_config,
                                            mapping=category_mapping)
    
    study = optuna.create_study(direction="minimize")
    
    n_trials = opt.trials

    # ========== DATA ========================================
    train_loader = pretraining_pipeline(os.path.join(opt.data, 'train'),
                                        batch_size= 5 if opt.debug else opt.bs, 
                                        window_size=opt.window_size,
                                        shuffle=True,
                                        sampling=True,
                                        repeat=4,
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
    
    # ===== MLFLOW SETUP ===================================
    run_name = opt.mlflow_run_name
    experiment_name = opt.mlflow_experiment_name
    tracking_uri = opt.mlflow_tracking_uri
    
    mf_set_tracking_uri(tracking_uri=tracking_uri)
    
    experiment_id = mf_create_or_get_experiment_id(experiment_name=experiment_name)
    
    mf_set_experiment(experiment_id=experiment_id)
    
    run_exists = mf_check_run_exists(run_name=run_name, experiment_name=experiment_name)
    
    if run_exists:
        print("""[INFO] The run with name {run_name} in experiment {experiment_name} already exists.
              It will be continued, if this is not the behaviour you expect,
              then please change the run name""".format(run_name=run_name, experiment_name=experiment_name))
        
    else:
        print("""[INFO] Starting new parent run with name {run_name} 
              for experiment {experiment_name}""".format(run_name=run_name, experiment_name=experiment_name))
        
    mf.start_run(run_name=run_name, experiment_id=experiment_id)
    best_astromer = None
    
    for number in range(n_trials):
        
        sweep_name = f"sweep_{number+1}"
        
        #Start a nested run for this trial
        
        mf.start_run(run_name=sweep_name, experiment_id=experiment_id, nested=True)
        trial = study.ask(distributions) 
        
        input_dict = trial.params
        
        ### Logging Model Configs 
        mf.log_params(input_dict)
        
        print(f"[INFO] The inputs to the get_ASTROMER function on trial {number+1} are: {input_dict}")
        print()
        model = get_ASTROMER(**input_dict,
                            pe_base=opt.pe_base,
                            pe_dim=opt.pe_dim,
                            pe_c=opt.pe_exp,
                            window_size=opt.window_size,
                            encoder_mode=opt.encoder_mode,
                            average_layers=opt.avg_layers,
                            mask_format=opt.mask_format)

    # ============================================================
        if opt.checkpoint != '-1':
            print('[INFO] Restoring previous training')
            model.load_weights(os.path.join(opt.checkpoint, 'weights', 'weights'))
            
        model, best_train_metrics, best_val_metrics, test_metrics = train(model,
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
                                                                argparse_dict=opt.__dict__,
                                                                scheduler=opt.scheduler)
        
        study.tell(trial, best_val_metrics["loss"])
        
        if best_val_metrics["loss"] <= study.best_value:
            best_astromer = model
        
        mf.end_run()
        
    mf.tensorflow.save_model(best_astromer,
                            path=f"{experiment_name}_{run_name}_astromer")
    
    mf.set_tags({"best_sweep":study.best_trial.number})
    
    mf.end_run()


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
    parser.add_argument('--avg-layers', action='store_true', help='If averaging outputs of the attention layers to form the final embedding. There is no avg if layers=1 ')
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
    
    
    ##Mlflow Args
    parser.add_argument('--trials', default=10, type=int, 
                        help='Number of Optuna trials')
    parser.add_argument('--optuna_config', default='presentation/experiments/config/', type=str, 
                        help='path to optuna conifg')
    parser.add_argument('--mlflow_run_name', default='AstromerIsCooking', type=str, 
                        help='Name of the mlflow parent run')
    parser.add_argument('--mlflow_experiment_name', default='Experiment_Astromer', type=str, 
                        help='Name of the mlflow experiment')
    parser.add_argument('--mlflow_tracking_uri', default='./runs', type=str, 
                        help='Name of the mlflow experiment')


    opt = parser.parse_args()        
    run(opt)
