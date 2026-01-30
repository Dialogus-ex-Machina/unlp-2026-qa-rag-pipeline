from .embeddings_model_spec import EmbeddingsModelSpec
from .embeddings_model_provider import EmbeddingsModelProvider
from langchain_core.embeddings import Embeddings


class EmbeddingsModelFactory:
    def __init__(self, providers: list[EmbeddingsModelProvider]):
        self._providers = providers

    def create(self, spec: EmbeddingsModelSpec) -> Embeddings:
        for _provider in self._providers:
            if _provider.can_create(spec):
                return _provider.create(spec)

        raise ValueError(f"Unsupported embeddings provider/spec: {spec}")
