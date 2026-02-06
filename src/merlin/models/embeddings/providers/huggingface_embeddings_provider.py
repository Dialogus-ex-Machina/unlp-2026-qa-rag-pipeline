from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from ...embeddings import (
    EmbeddingsProvider,
    EmbeddingsSpec
)


class HuggingFaceEmbeddingsProvider(EmbeddingsProvider):
    def can_create(self, spec: EmbeddingsSpec) -> bool:
        return spec.provider == "huggingface" and not (spec.model_name or "").isspace()

    def create(self, spec: EmbeddingsSpec) -> Embeddings:
        return HuggingFaceEmbeddings(
            model_name=spec.model_name,
            cache_folder=spec.cache_folder,
            model_kwargs=spec.model_kwargs,
            encode_kwargs=spec.encode_kwargs,
            query_encode_kwargs=spec.query_encode_kwargs,
            multi_process=spec.multi_process,
            show_progress=spec.show_progress
        )

