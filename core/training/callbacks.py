import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

def get_callbacks(project_dir):
    esp_callback = EarlyStopping(monitor ='val_loss',
                                 mode = 'min',
                                 patience = opt.patience,
                                 restore_best_weights=True)
    tsb_callback = TensorBoard(
                    log_dir = os.path.join(project_dir, 'logs'),
                    histogram_freq=1,
                    write_graph=True)

    hvd_callback = hvd.callbacks.BroadcastGlobalVariablesCallback(0)

    return [ckp_callback, esp_callback, tsb_callback, hvd_callback]
