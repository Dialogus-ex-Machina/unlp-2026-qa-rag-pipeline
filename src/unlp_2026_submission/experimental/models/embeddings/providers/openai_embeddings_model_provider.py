from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from ...embeddings import (
    EmbeddingsModelProvider,
    EmbeddingsModelSpec
)


class OpenAIEmbeddingsModelProvider(EmbeddingsModelProvider):
    def can_create(self, spec: EmbeddingsModelSpec) -> bool:
        return spec.provider == "openai" and (spec.model_name or "").lower().startswith("text-embedding")

    def create(self, spec: EmbeddingsModelSpec) -> Embeddings:
        return OpenAIEmbeddings(
            model=spec.model_name,
            api_key=spec.api_key
        )