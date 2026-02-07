from typing import Protocol
from langchain_core.embeddings import Embeddings

from ..embeddings import EmbeddingsSpec


class EmbeddingsProvider(Protocol):
    def can_create(self, spec: EmbeddingsSpec) -> bool: ...
    def create(self, spec: EmbeddingsSpec) -> Embeddings: ...
