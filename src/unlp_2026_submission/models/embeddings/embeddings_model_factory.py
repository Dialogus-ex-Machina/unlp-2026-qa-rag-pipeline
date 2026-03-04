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
            embeddings_model = OpenAIEmbeddingsModel.create(
                model_name=self._config.embeddings_model_name,
                api_key=self._config.model_provider_api_key,
            )
            return embeddings_model

        if GoogleEmbeddingsModel.is_compatible_model(self._config.embeddings_model_name):
            embeddings_model = GoogleEmbeddingsModel.create(
                model_name=self._config.embeddings_model_name,
                api_key=self._config.model_provider_api_key,
            )

            return embeddings_model

        embeddings_model = SentenceTransformerEmbeddingModel.create(
            model_name=self._config.embeddings_model_name,
            cache_dir=self._config.downloaded_models_cache_dir,
        )
        return embeddings_model
