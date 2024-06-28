from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Input,
)
from tensorflow.keras.models import Model


class AttentionModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        x = Concatenate(axis=-1)([inputs])
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def compile(self):
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)

    def predict(self, x):
        return self.model.predict(x)
