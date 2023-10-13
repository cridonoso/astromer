#!/usr/bin/env python

import pandas as pd
import os, sys
import tensorflow as tf
from src.data import pretraining_pipeline, load_data
from src.models.astromer_1 import get_ASTROMER, train_step, test_step
from presentation.experiments.utils import train_classifier
from src.models.astromer_1 import get_ASTROMER, build_input, train_step, test_step
from src.training.utils import train
import time
import argparse
import toml
from tensorflow.keras.callbacks  import (ModelCheckpoint,
                                         EarlyStopping,
                                         TensorBoard)



def run(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    start_time = time.time()
    with open(os.path.join(opt.config_path), 'r') as f:
        model_config = toml.load(f)
        
    BATCH_SIZE = opt.bs
    print('BATCH_SIZE', BATCH_SIZE)

    astromer = get_ASTROMER(num_layers=model_config['num_layers'],
                num_heads=model_config['num_heads'],
                head_dim=model_config['head_dim'],
                mixer_size=model_config['mixer'],
                dropout=model_config['dropout'],
                pe_base=model_config['pe_base'],
                pe_dim=model_config['pe_dim'],
                pe_c=model_config['pe_exp'],
                window_size=model_config['window_size'],
                encoder_mode=model_config['encoder_mode'],
                average_layers=model_config['avg_layers'],
                batch_size=BATCH_SIZE)

    train_batches = load_data(dataset='{}/macho_pt_train/train'.format(opt.data_path),
                                        batch_size=BATCH_SIZE,
                                        window_size=200,
                                        probed=model_config['probed'], 
                                        random_same=model_config['rs'],
                                        sampling=False,
                                        off_nsp=True, 
                                        repeat=4)
    valid_batches = load_data(dataset='{}/fold_0/macho_pt_val/val'.format(opt.data_path),
                                        batch_size=BATCH_SIZE,
                                        window_size=200,
                                        off_nsp=True, 
                                        probed=model_config['probed'],
                                        random_same=model_config['rs'],
                                        sampling=False,
                                        repeat=1)
    test_loader = load_data(dataset='{}/fold_0/macho_pt_val/test'.format(opt.data_path), 
                                batch_size=BATCH_SIZE, 
                                probed=model_config['probed'],  
                                random_same=model_config['rs'],
                                window_size=200, 
                                off_nsp=True, 
                                repeat=1, 
                                sampling=False)
    # TRAIN ASTROMER
    astromer, (best_train_log, best_val_log, test_metrics), callback = train(astromer,
                train_batches, 
                valid_batches, 
                num_epochs=10000, 
                lr= 1e-5, 
                test_loader=test_loader,
                project_path=opt.pt_folder,
                debug=False,
                patience=20,
                train_step_fn=train_step,
                test_step_fn=test_step )

    fold=0
    print('Resultados', (best_train_log, best_val_log, test_metrics))
    best_train_log['step'] = 'train'
    best_val_log['step'] = 'val'
    test_metrics['step'] = 'test'
    best_val_log['fold'] = fold
    best_train_log['fold'] = fold
    test_metrics['fold'] = fold

    print('gpu_max ',max(callback) )
    data = [best_train_log, best_val_log, test_metrics]
    df = pd.DataFrame(data)
    ruta_archivo_csv = "{}/metricas.csv".format(opt.pt_folder)

    if os.path.exists(ruta_archivo_csv):
        print(f"El archivo '{ruta_archivo_csv}' existe")
        resultado = pd.read_csv(ruta_archivo_csv)
        resultado = pd.concat([df, resultado], axis=0)
        resultado.to_csv(ruta_archivo_csv,index=False)
    else:
        print(f"El archivo '{ruta_archivo_csv}' no existe")
        df.to_csv(ruta_archivo_csv, index = False)

    elapsed_time = (time.time() - start_time) / 60
    print(f'Tiempo transcurrido: {elapsed_time:.2f} minutos')

    elapsed_time = (time.time() - start_time) / 3600
    print(f'Tiempo transcurrido: {elapsed_time:.2f} horas')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default='-1', type=str, help='GPU to be used. -1 means no GPU will be used')
	parser.add_argument('--data-path', default='alcock', type=str)
	parser.add_argument('--pt-folder', default='./results/pretraining*', type=str, help='pretrained model folder')
	parser.add_argument('--config-path', default='./model_config.toml', type=str, help='config model folder')

	parser.add_argument('--bs', default=2000, type=int,	help='Batch size')
	parser.add_argument('--patience', default=20, type=int,	help='Earlystopping threshold in number of epochs')
	parser.add_argument('--num_epochs', default=10000, type=int, help='Number of epochs')

	opt = parser.parse_args()        
	run(opt)
