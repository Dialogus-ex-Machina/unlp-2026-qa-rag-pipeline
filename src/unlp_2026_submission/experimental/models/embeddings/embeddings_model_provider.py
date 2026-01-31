from typing import Protocol
from langchain_core.embeddings import Embeddings

from ..embeddings import EmbeddingsModelSpec


class EmbeddingsModelProvider(Protocol):
    def can_create(self, spec: EmbeddingsModelSpec) -> bool: ...
    def create(self, spec: EmbeddingsModelSpec) -> Embeddings: ...
