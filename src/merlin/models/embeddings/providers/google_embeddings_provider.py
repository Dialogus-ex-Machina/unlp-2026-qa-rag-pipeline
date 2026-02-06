from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings

from ...embeddings import (
    EmbeddingsProvider,
    EmbeddingsSpec
)


class GoogleEmbeddingsProvider(EmbeddingsProvider):
    def can_create(self, spec: EmbeddingsSpec) -> bool:
        return spec.provider == "google" and (spec.model_name or "").lower().startswith("gemini-embedding")

    def create(self, spec: EmbeddingsSpec) -> Embeddings:
        return GoogleGenerativeAIEmbeddings(
            model=spec.model_name,
            api_key=spec.api_key
        )