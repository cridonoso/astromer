import tensorflow as tf

from .classifier import Classifier
from .feature_structure_module import FeatureStructureModule


class MultiBandAstromerSystem:
    def __init__(
        self,
        pt_paths: list[str],
        output_paths: list[str],
        input_shape: tuple[int],
        num_classes: int,
    ) -> None:
        self.feature_module = FeatureStructureModule(pt_paths, output_paths)
        self.attention_model = AttentionModel(input_shape, num_classes)

    def generate_attention_vectors(
        self,
        data_opts: list[dict]
    ) -> None:
        self.feature_module.generate_attention_vectors(data_opts)

    def train_attention_model(
        self,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        self.attention_model.compile()
        self.attention_model.train(x_train, y_train, epochs, batch_size)

    def save_attention_model(self, filename: str) -> None:
        self.attention_model.save_model(filename)

    def load_attention_model(self, filename: str) -> None:
        self.attention_model.load_model(filename)

    def predict_with_attention_model(self, x: tf.Tensor) -> tf.Tensor:
        return self.attention_model.predict(x)
