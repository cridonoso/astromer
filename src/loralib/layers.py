import tensorflow as tf
from tensorflow.keras.layers import Dropout 
from typing import Optional

class LoRALayer(tf.Module):
    def __init__(
            self, 
            r: int, #rank
            lora_alpha: int, #learning_rate
            lora_dropout: float, #dropout 
            merge_weights: bool, #boolean flag to merge weights
    ):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        #dropout optional 
        if lora_dropout > 0.:
            self.lora_dropout = Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        #mark the weight as unmerged
        self.merged = False 
        self.merge_weights = merge_weights


class Embedding(tf.keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, r, lora_alpha, merge_weights=True):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.merge_weights = merge_weights
        self.lora_alpha = lora_alpha
        self.embeddings = tf.Variable(
            tf.random.normal([num_embeddings, embedding_dim]),
            trainable=True,
            name='embeddings')
        self.r = r

        # Low-Rank Approximation
        self.lora_A = tf.Variable(
            tf.random.normal([num_embeddings, r]),
            trainable=True,
            name='lora_A')
        self.lora_B = tf.Variable(
            tf.random.normal([r, embedding_dim]),
            trainable=True,
            name='lora_B')

        self.merged = False
        self.scaling = tf.math.sqrt(tf.cast(embedding_dim, dtype=tf.float32))

    def reset_parameters(self):
        """Resets the LoRA parameters to their initial values."""
        if self.r > 0:
            self.lora_A.assign(tf.zeros_like(self.lora_A))
            self.lora_B.assign(tf.zeros_like(self.lora_B))
            self.merged = False

    def build(self, input_shape):
        pass

    def call(self, x):
        if self.r > 0 and not self.merged:
            frequent_tokens = ...  # Define your top-r tokens based on your dataset
            lora_embeddings = tf.nn.embedding_lookup(tf.transpose(self.lora_A), frequent_tokens)  # Shape: [batch_size, sequence_len, r]
            lora_embeddings = tf.matmul(lora_embeddings, self.lora_B)  # We use matmul 
            lora_embeddings = tf.reshape(lora_embeddings, tf.shape(result))  # Ensure lora_embeddings has the same shape as result
            lora_embeddings *= self.scaling
        else:
            lora_embeddings = 0
        result = tf.nn.embedding_lookup(tf.transpose(self.embeddings), x) + lora_embeddings
        return result

    def train(self, training=True):
        self.trainable = training
        if not training and self.merge_weights and not self.merged:
            self.embeddings.assign_add(tf.tensordot(self.lora_A, self.lora_B, axes=[[0], [0]]) * self.scaling)
            self.merged = True
        elif training:
            self.merged = False
            if hasattr(self, 'lora_A'):
                self.lora_A.assign(tf.zeros_like(self.lora_A))
            if hasattr(self, 'lora_B'):
                self.lora_B.assign(tf.zeros_like(self.lora_B))




        
