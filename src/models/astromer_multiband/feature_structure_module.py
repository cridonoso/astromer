import numpy as np

from presentation.pipelines.steps.model_design import load_pt_model
from src.data.loaders import get_loader


class FeatureStructureModule:
    def __init__(self, pt_paths: list[str], output_paths: list[str]) -> None:
        assert len(pt_paths) == len(output_paths)
        self.models = []
        self.output_paths = output_paths
        self._load_pretrained_model_encoders(pt_paths)

    def _create_data_loader(self, data_opt: list[dict]) -> None:
        self.datasets = []
        for opt in data_opt:
            self.datasets.append(get_loader(**opt))

    def _load_pretrained_model_encoders(self, pt_paths: list[str]) -> None:
        for pt_path in pt_paths:
            model, _ = load_pt_model(pt_path)
            encoder = model.get_layer('encoder')
            encoder.trainable = False
            self.models.append(encoder)

    def _compute_attention_vectors(self) -> None:
        for i, dataset in enumerate(self.datasets):
            attention_vectors = []
            model = self.models[i]
            for batch in dataset:
                output = model(batch)
                attention_vectors.append(output.numpy())
            np.save(self.output_paths[i], np.concatenate(attention_vectors))

    def generate_attention_vectors(self, data_opts: list[dict]) -> None:
        self._create_data_loader(data_opts)
        self._compute_attention_vectors()
