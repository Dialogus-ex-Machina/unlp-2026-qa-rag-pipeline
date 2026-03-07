from unlp_2026_submission.config import Config
from langchain_core.embeddings import Embeddings as EmbeddingsModel
from .sentence_transformer_embeddings_model import SentenceTransformerEmbeddingModel

class EmbeddingsModelFactory:
    _config: Config

    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def create(config: Config):
        return EmbeddingsModelFactory(config)

    def get_embeddings_model(self) -> EmbeddingsModel:
        embeddings_model = SentenceTransformerEmbeddingModel.create(
            model_name_or_path=self._config.embeddings_model_name,
            cache_dir=self._config.downloaded_models_cache_dir,
        )
        return embeddings_model
