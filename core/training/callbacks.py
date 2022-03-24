import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

def get_callbacks(project_dir):
    ckp_callback = ModelCheckpoint(
                    filepath=os.path.join(project_dir, 'weights.h5'),
                    save_weights_only=True,
                    monitor='val_loss',
                    mode='min',
                    save_best_only=True)
    esp_callback = EarlyStopping(monitor ='val_loss',
                                 mode = 'min',
                                 patience = opt.patience,
                                 restore_best_weights=True)
    tsb_callback = TensorBoard(
                    log_dir = os.path.join(project_dir, 'logs'),
                    histogram_freq=1,
                    write_graph=True)

    return [ckp_callback, esp_callback, tsb_callback]
