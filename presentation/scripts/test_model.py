import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import toml
import sys
import os

from sklearn.metrics import r2_score, root_mean_squared_error
from src.utils import get_metrics
from presentation.pipelines.steps.model_design import load_pt_model
from presentation.pipelines.steps.load_data import build_loader 

def compute_metrics(output):
    y = tf.ragged.boolean_mask(output['magnitudes'], output['probed_mask'])
    y_hat = tf.ragged.boolean_mask(output['reconstruction'], output['probed_mask'])
    
    r2_values = []
    mse_values = []
    for i in range(output['magnitudes'].shape[0]):
        y = tf.boolean_mask(output['magnitudes'][i], output['probed_mask'][i])
        y_hat = tf.boolean_mask(output['reconstruction'][i], output['probed_mask'][i])
        y = y.numpy()
        y_hat = y_hat.numpy()
        print(y.shape, y_hat.shape)
        
        r2_values.append(r2_score(y, y_hat))
        mse_values.append(root_mean_squared_error(y, y_hat))
        
    test_r2   = np.mean(r2_values) 
    test_mse = np.mean(mse_values)
    
    return test_r2, test_mse

def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    astromer, config = load_pt_model(opt.model)
    df = pd.DataFrame(config, index=[0])
    loaders = build_loader(df['data'].values[0], 
                           config, 
                           batch_size=opt.bs,
                           clf_mode=False,
                           sampling=False,
                           return_test=True,
                           normalize='zero-mean')     
    output = astromer.predict(loaders['test'])
    r2_value, mse_value = compute_metrics(output)
    
    valid_loss = get_metrics(os.path.join(opt.model, 'tensorboard', 'validation'), 
                                metric_name='epoch_loss')
    
    best_loss = valid_loss[valid_loss['value']==valid_loss['value'].min()]

    valid_rsquare = get_metrics(os.path.join(opt.model, 'tensorboard', 'validation'), 
                                metric_name='epoch_r_square')
    
    metrics = {
        'test_r2': [r2_value],
        'test_mse': [mse_value],
        'val_mse': [float(best_loss['value'].values[0])],
        'val_r2': [float(valid_rsquare.iloc[best_loss.index]['value'].values[0])]
    }
    metrics = pd.DataFrame(metrics)
    df = pd.concat([df, metrics], axis=1)
    df.to_csv(os.path.join(opt.model, 'results.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/presentation/results/model', type=str,
                    help='Model folder')
    parser.add_argument('--gpu', default='-1', type=str,
                        help='GPU Device')
    parser.add_argument('--bs', default=1024, type=int,
                        help='Batch size')



    opt = parser.parse_args()        
    run(opt)
