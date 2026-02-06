from .embeddings_spec import EmbeddingsSpec
from .embeddings_provider import EmbeddingsProvider
from langchain_core.embeddings import Embeddings

from .providers import (
    GoogleEmbeddingsProvider,
    OpenAIEmbeddingsProvider,
    HuggingFaceEmbeddingsProvider,
)


class EmbeddingsFactory:
    def __init__(self, providers: list[EmbeddingsProvider]):
        self._providers = providers

    def create(self, spec: EmbeddingsSpec) -> Embeddings:
        for _provider in self._providers:
            if _provider.can_create(spec):
                return _provider.create(spec)

        raise ValueError(f"Unsupported embeddings provider/spec: {spec}")

    @staticmethod
    def create_all_embeddings_factory():
        return EmbeddingsFactory([
            GoogleEmbeddingsProvider(),
            OpenAIEmbeddingsProvider(),
            HuggingFaceEmbeddingsProvider(),
        ])
