from unlp_2026_submission.config import Config
from unlp_2026_submission.models.embeddings import (
    OpenAIEmbeddingsModel,
    EmbeddingsModel,
    GoogleEmbeddingsModel,
    SentenceTransformerEmbeddingModel
)

class EmbeddingsModelFactory:
    _config: Config

    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def create(config: Config):
        return EmbeddingsModelFactory(config)

    def get_embeddings_model(self) -> EmbeddingsModel:
        if OpenAIEmbeddingsModel.is_compatible_model(self._config.embeddings_model_name):
            embeddings_model = OpenAIEmbeddingsModel.create(self._config)
            return embeddings_model

        if GoogleEmbeddingsModel.is_compatible_model(self._config.embeddings_model_name):
            embeddings_model = GoogleEmbeddingsModel.create(self._config)

            return embeddings_model

        embeddings_model = SentenceTransformerEmbeddingModel.create(self._config)
        return embeddings_model
