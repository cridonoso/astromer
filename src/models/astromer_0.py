import tensorflow as tf

from src.losses              import custom_rmse
from src.metrics             import custom_r2
from tensorflow.keras.layers import Input
from tensorflow.keras        import Model

from src.layers.positional import positional_encoding

from tensorflow.keras.layers import Input, Layer, Dense
import tensorflow as tf


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    steps = tf.shape(scaled_attention_logits)[2]
    mask_rshp = tf.tile(mask, [1,1,steps])
    mask_rshp += tf.transpose(mask_rshp, [0,2,1])
    mask_rshp = tf.minimum(1., mask_rshp)
    mask_rshp = tf.expand_dims(mask_rshp, 1)
    scaled_attention_logits += mask_rshp

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1, name='MaskedSoftMax')  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v, name='Z')  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads # final dimension

        self.wq = tf.keras.layers.Dense(d_model, name='WQ')
        self.wk = tf.keras.layers.Dense(d_model, name='WK')
        self.wv = tf.keras.layers.Dense(d_model, name='WV')

        self.dense = tf.keras.layers.Dense(d_model, name='MixerDense')

    def split_heads(self, x, batch_size, name='qkv'):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3], name=name)

    def call(self, x, mask):
        batch_size = tf.shape(x)[0]

        q = self.wq(x)  # (batch_size, seq_len, d_model)
        k = self.wk(x)  # (batch_size, seq_len, d_model)
        v = self.wv(x)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, name='Q')  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, name='K')  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, name='V')  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class RegLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reg_layer = Dense(1, name='RegLayer')
        self.bn_0 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.bn_0(inputs)
        x = self.reg_layer(x)
        return x
    
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='tanh'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, use_leak=False, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.use_leak = use_leak
        if use_leak:
            self.reshape_leak_1 = tf.keras.layers.Dense(d_model)
            self.reshape_leak_2 = tf.keras.layers.Dense(d_model)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        if self.use_leak:
            out1 = self.layernorm1(self.reshape_leak_1(x) + attn_output)  # (batch_size, input_seq_len, d_model)
        else:
            out1 = self.layernorm1(attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        if self.use_leak:
            out2 = self.layernorm2(self.reshape_leak_2(out1) + ffn_output) # (batch_size, input_seq_len, d_model)
        else:
            out2 = self.layernorm2(ffn_output)

        return out2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 base=10000, rate=0.1, use_leak=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.base = base
        self.inp_transform = tf.keras.layers.Dense(d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, use_leak)
                            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, data, training=False):
        # adding embedding and position encoding.
        x_pe = positional_encoding(data['times'], self.d_model, mjd=True)

        x_transformed = self.inp_transform(data['input'])
        transformed_input = x_transformed + x_pe

        x = self.dropout(transformed_input, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, data['mask_in'])

        return x  # (batch_size, input_seq_len, d_model)
    
def build_input(length):
    serie  = Input(shape=(length, 1),
                  batch_size=None,
                  name='input')
    times  = Input(shape=(length, 1),
                  batch_size=None,
                  name='times')
    mask   = Input(shape=(length, 1),
                  batch_size=None,
                  name='mask')

    return {'input':serie,
            'mask_in':mask,
            'times':times}

def get_ASTROMER(num_layers=2,
                 d_model=200,
                 num_heads=2,
                 dff=256,
                 base=10000,
                 rate=0.1,
                 use_leak=False,
                 maxlen=100,
                 batch_size=None):

    placeholder = build_input(maxlen)

    encoder = Encoder(num_layers,
                      d_model,
                      num_heads,
                      dff,
                      base=base,
                      rate=rate,
                      use_leak=False,
                      name='encoder')

    x = encoder(placeholder)

    x = RegLayer(name='regression')(x)

    return CustomModel(inputs=placeholder, outputs=x, name="ASTROMER")


# @tf.function
# def train_step(model, x, y, optimizer, **kwargs):
#     with tf.GradientTape() as tape:
#         x_pred = model(x, training=True)
#         rmse = custom_rmse(y_true=y['target'],
#                           y_pred=x_pred,
#                           mask=y['mask_out'])
#         r2_value = custom_r2(y['target'], x_pred, y['mask_out'])

#     grads = tape.gradient(rmse, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}


# @tf.function
# def test_step(model, x, y, return_pred=False, **kwargs):
#     x_pred = model(x, training=False)
#     rmse = custom_rmse(y_true=y['target'],
#                       y_pred=x_pred,
#                       mask=y['mask_out'])
#     r2_value = custom_r2(y['target'], x_pred, y['mask_out'])
#     if return_pred:
#         return x_pred
#     return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}

# def predict(model, test_loader):
#     n_batches = sum([1 for _, _ in test_loader])
#     print('[INFO] Processing {} batches'.format(n_batches))
#     y_pred = tf.TensorArray(dtype=tf.float32, size=n_batches)
#     y_true = tf.TensorArray(dtype=tf.float32, size=n_batches)
#     masks  = tf.TensorArray(dtype=tf.float32, size=n_batches)
#     times  = tf.TensorArray(dtype=tf.float32, size=n_batches)
#     for index, (x, y) in enumerate(test_loader):
#         outputs = test_step(model, x, y, return_pred=True)
#         y_pred = y_pred.write(index, outputs)
#         y_true = y_true.write(index, y['target'])
#         masks  = masks.write(index, y['mask_out'])
#         times = times.write(index, x['times'])

#     y_pred = tf.concat([times.concat(), y_pred.concat()], axis=2)
#     y_true = tf.concat([times.concat(), y_true.concat()], axis=2)
#     return y_pred, y_true, masks.concat()

# def restore_model(model_folder):
#     with open(os.path.join(model_folder, 'config.toml'), 'r') as f:
#         model_config = toml.load(f)


#     astromer = get_ASTROMER(num_layers=model_config['num_layers'],
#                             d_model=model_config['head_dim']*model_config['num_heads'],
#                             num_heads=model_config['num_heads'],
#                             dff=model_config['mixer'],
#                             base=model_config['pe_base'],
#                             rate=model_config['dropout'],
#                             use_leak=False,
#                             maxlen=model_config['window_size'])

#     print('[INFO] LOADING PRETRAINED WEIGHTS')
#     astromer.load_weights(os.path.join(model_folder, 'weights', 'weights'))

#     return astromer, model_config


### KERAS MODEL 
class CustomModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            x_pred = self(x, training=True)
            rmse = custom_rmse(y_true=y['target'],
                              y_pred=x_pred,
                              mask=y['mask_out'])
            r2_value = custom_r2(y['target'], x_pred, y['mask_out'])

        grads = tape.gradient(rmse, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}

    def test_step(self, data):
        x, y = data
        x_pred = self(x, training=False)
        rmse = custom_rmse(y_true=y['target'],
                          y_pred=x_pred,
                          mask=y['mask_out'])
        r2_value = custom_r2(y['target'], x_pred, y['mask_out'])
        return {'loss': rmse, 'r_square':r2_value, 'rmse':rmse}
    
    def predict_step(self, data):
        x, y = data
        x_pred = self(x, training=False)
        
        return {'reconstruction': x_pred, 
                'magnitudes': y['target'],
                'times': x['times'],
                'probed_mask': y['mask_out']}