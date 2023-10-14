import keras

class LoraLayer(keras.layers.Layer):
    def __init__(
        self,
        original_layer,
        output_shape,
        name,
        rank=8,
        alpha=32,
        trainable=False,
        **kwargs,):


        super().__init__(name=name, trainable=trainable, **kwargs)

        self.rank = rank
        self.alpha = alpha

        self._scale = alpha / rank
        self.outputshape = output_shape
        self._num_heads = self.outputshape[-3]
        self._hidden_dim = self._num_heads * self.outputshape[-1]

        self.original_layer = original_layer
        self.original_layer.trainable = False
        
        self.A = keras.layers.Dense(
            units=rank,
            use_bias=False,
            kernel_initializer=keras.initializers.VarianceScaling(
                scale=math.sqrt(5), mode="fan_in", distribution="uniform"
            ),
            trainable=trainable,
            name=f"lora_A",
        )

        self.B = keras.layers.EinsumDense(
            equation='abc,cde->abde',
            output_shape=self.outputshape,
            kernel_initializer="zeros",
            trainable=trainable,
            name=f"lora_B",
        )

    def call(self, inputs):
        original_output = self.original_layer(inputs)
        if self.trainable:
            lora_output = self.B(self.A(inputs)) * self._scale
            return original_output + lora_output
        return original_output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'rank': self.rank,
            'alpha': self.alpha,
            'trainable': self.trainable,
            'name': self.name
        })
        return config