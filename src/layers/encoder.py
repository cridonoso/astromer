import tensorflow as tf

from src.layers.attention import HeadAttentionMulti
from src.layers.positional import positional_encoding, PositionalEncoder

from tensorflow.keras import Model

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='tanh'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, head_dim, num_heads, dff, dropout=0.1, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout
        self.d_model = self.head_dim*self.num_heads

    def build(self, input_shape):
        self.mha = HeadAttentionMulti(self.head_dim, self.num_heads)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout)

    def call(self, x, training=False, mask=None, return_weights=False):
        attn_output, att_weights = self.mha(x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = self.layernorm1(attn_output)
        ffn_output  = self.ffn(attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output  = self.dropout2(ffn_output, training=training)
        ffn_output  = self.layernorm2(ffn_output)
        if return_weights:
            return ffn_output, att_weights
        return ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout": self.rate,
        })
        return config

class Encoder(Model):
    def __init__(self, 
                 window_size,
                 num_layers, 
                 num_heads, 
                 head_dim, 
                 mixer_size=128,
                 dropout=0.1, 
                 pe_base=1000, 
                 pe_dim=128,
                 pe_c=1., 
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.window_size = window_size
        self.num_layers  = num_layers
        self.num_heads   = num_heads
        self.head_dim    = head_dim
        self.mixer_size  = mixer_size
        self.dropout     = dropout
        self.pe_base     = pe_base
        self.pe_c        = pe_c
        self.pe_dim      = pe_dim

        self.positional_encoder = PositionalEncoder(self.pe_dim, base=self.pe_base, c=self.pe_c, name='PosEncoding')
        
        self.enc_layers = [AttentionBlock(self.head_dim, self.num_heads, self.mixer_size, dropout=self.dropout)
                            for _ in range(self.num_layers)]

    def call(self, data, training=False):
        # adding embedding and position encoding.
        x_pe = self.positional_encoder(data['times'])
        x = tf.concat([x_pe, data['magnitudes'], data['seg_emb']], 2)
        
        layers_outputs = []
        for i in range(self.num_layers):
            z =  self.enc_layers[i](x, mask=data['att_mask'])
            layers_outputs.append(z)
        x = tf.reduce_mean(layers_outputs, 0)
        x = tf.reshape(x, [-1, self.window_size+1, self.num_heads*self.head_dim])
        return   x # (batch_size, input_seq_len, d_model)