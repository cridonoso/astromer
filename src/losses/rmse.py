import tensorflow as tf

@tf.function
def rmse_for_nsp(y_true, y_pred, mask=None, nsp_label=None, segment_emb=None):
    inp_shp = tf.shape(y_true)
    residuals = tf.square(y_true - y_pred)

    segment_emb = segment_emb*(1.-tf.expand_dims(nsp_label, 1))

    loss_first_50 = (residuals*mask) * segment_emb
    loss_all = (residuals * mask) * tf.expand_dims(nsp_label, 1)
    loss_total = loss_first_50 + loss_all

    N = tf.where(loss_total == 0., 0., 1.)
    N = tf.reduce_sum(N, axis=1)
    
    mse_mean = tf.math.divide_no_nan(tf.reduce_sum(loss_total, 1), N)
    mse_mean = tf.reduce_mean(mse_mean)
    return mse_mean

@tf.function
def custom_rmse(y_true, y_pred, mask=None):
    inp_shp = tf.shape(y_true)
    
    residuals = tf.square(y_true - y_pred)
    residuals = tf.multiply(residuals, mask)
    residuals  = tf.reduce_sum(residuals, 1)
    mse_mean = tf.math.divide_no_nan(residuals, tf.reduce_sum(mask, 1))
    mse_mean = tf.reduce_mean(mse_mean)
    return mse_mean