import tensorflow as tf

from src.layers.attention import HeadAttentionMulti
from src.layers.positional import positional_encoding, PositionalEncoder
from src.data import reshape_mask

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='tanh'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, use_leak=False, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.use_leak = use_leak
        # = ======================== = ======================== =
        self.mha = HeadAttentionMulti(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.use_leak = use_leak
        if use_leak:
            self.reshape_leak_1 = tf.keras.layers.Dense(d_model)
            self.reshape_leak_2 = tf.keras.layers.Dense(d_model)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
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

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate,
            "use_leak": self.use_leak
        })
        return config

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 base=10000, dropout=0.1, pe_c=1., **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model    = d_model
        self.num_layers = num_layers
        self.base       = base
        self.dropout    = dropout

        self.pe = PositionalEncoder(d_model, base=base, c=pe_c, name='PosEncoding')
        self.inp_transform = tf.keras.layers.Dense(d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout, False)
                            for _ in range(num_layers)]
        self.dropout_layer = tf.keras.layers.Dropout(dropout)

    def call(self, data, training=False):
        # adding embedding and position encoding.
        x_pe = self.pe(data['times'])
        x_transformed = self.inp_transform(data['input'])
        transformed_input = x_transformed + x_pe

        x = self.dropout_layer(transformed_input, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, data['mask_in'])

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "base": self.base,
            "dropout": self.dropout
        })
        return config

class EncoderSKIP(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 base=10000, dropout=0.1, use_leak=False, pe_v2=False, **kwargs):
        super(EncoderSKIP, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.base = base
        self.pe_v2 = pe_v2
        self.inp_transform = tf.keras.layers.Dense(d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout, use_leak)
                            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, data, training=False):
        # adding embedding and position encoding.
        x_pe = positional_encoding(data['times'],
                                   self.d_model,
                                   base=self.base,
                                   mjd=True,
                                   v2=self.pe_v2)

        x_transformed = self.inp_transform(data['input'])
        transformed_input = x_transformed + x_pe

        x = self.dropout(transformed_input, training=training)

        layers_outputs = []
        for i in range(self.num_layers):
            z =  self.enc_layers[i](x, mask=data['mask_in'])
            layers_outputs.append(z)

        x = tf.reduce_mean(layers_outputs, 0)

        return x  # (batch_size, input_seq_len, d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "base": self.base,
            "pe_v2": self.pe_v2,
            "dropout": self.dropout
        })
        return config
