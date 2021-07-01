import tensorflow as tf

from core.attention import MultiHeadAttention
from core.input     import input_format
from core.positional import positional_encoding, ShuklaEmbedding
from core.masking import reshape_mask

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.reshape_leak_1 = tf.keras.layers.Dense(d_model)
        self.reshape_leak_2 = tf.keras.layers.Dense(d_model)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)



    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(self.reshape_leak_1(x) + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(self.reshape_leak_2(out1) + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2

def segment_embedding(x_transformed, seplim):
    tensor_shape = tf.shape(x_transformed)
    def fn(x):
        seplim = tf.cast(x, tf.int32)
        zeros = tf.zeros([tensor_shape[1]-seplim, tensor_shape[-1]])
        ones = tf.ones([seplim, tensor_shape[-1]])
        pe = tf.concat([zeros, ones], 0)
        return pe
    input_pe = tf.map_fn(lambda x: fn(x), seplim)
    x_transformed+=input_pe
    return x_transformed

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 base=10000, rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.base = base
        self.inp_transform = tf.keras.layers.Dense(d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        # self.pe_emb = ShuklaEmbedding(dim_model=d_model)
    def call(self, data, training=False):
        # Reshape MASK
        mask = reshape_mask(data['mask_in']) # batch x 1 x seq_len x seq_len
        # adding embedding and position encoding.
        x_pe = positional_encoding(data['times'], self.d_model, mjd=True)
        # x_pe = self.pe_emb(data['times'])

        x_transformed = self.inp_transform(data['input'])

        transformed_input = x_transformed + x_pe
        x = self.dropout(transformed_input, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)
