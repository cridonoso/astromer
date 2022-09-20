import tensorflow as tf
import optuna

from tensorflow.keras.callbacks import (ModelCheckpoint,
                                        EarlyStopping,
                                        TensorBoard)
from core.data                  import pretraining_pipeline
from core.training              import CustomSchedule
from core.models                import get_ASTROMER

import sys, os

print(sys.argv[1])
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

def objective(trial):

    datapath = './data/records/macho'

    num_layers = trial.suggest_int("num_layers", 1, 8)
    head_dim = trial.suggest_categorical("head_dim", [2, 4, 8, 16])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16, 32])
    dff = trial.suggest_int("dff", 64, 512)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    astromer = get_ASTROMER(num_layers=num_layers,
                            d_model=head_dim*num_heads,
                            num_heads=num_heads,
                            dff=dff,
                            base=1000,
                            rate=dropout,
                            maxlen=200)

    lrate = CustomSchedule(head_dim*num_heads)
    optimizer = tf.keras.optimizers.Adam(lrate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)
    astromer.compile(optimizer=optimizer)


    train_batches = pretraining_pipeline(os.path.join(datapath, 'train'),
                                         batch_size=256,
                                         shuffle=True,
                                         repeat=1,
                                         cache=True,
                                         window_size=200,
                                         sampling=True,
                                         msk_frac=.5,
                                         rnd_frac=.2,
                                         same_frac=.2,
                                         per_sample_mask=True)
    valid_batches = pretraining_pipeline(os.path.join(datapath, 'val'),
                                         batch_size=256,
                                         shuffle=False,
                                         repeat=1,
                                         cache=True,
                                         window_size=200,
                                         sampling=True,
                                         msk_frac=.5,
                                         rnd_frac=.2,
                                         same_frac=.2,
                                         per_sample_mask=True)

    callbacks = [
            EarlyStopping(monitor ='val_loss',
                          mode = 'min',
                          patience = 20,
                          restore_best_weights=True)
                          ]

    # Training
    hist = astromer.fit(train_batches,
                        epochs=1000,
                        validation_data=valid_batches,
                        callbacks=callbacks)

    min_loss = min(hist.history['val_loss'])
    max_r2   = max(hist.history['val_r_square'])

    return min_loss, max_r2


if __name__ == "__main__":
    study = optuna.load_study(
        study_name="astromer_hp",
        storage="mysql://root:abcdabcd@astromer.c1hqeh36ya5n.us-east-1.rds.amazonaws.com:3306/hp"
    )
    study.optimize(objective, n_trials=100)
