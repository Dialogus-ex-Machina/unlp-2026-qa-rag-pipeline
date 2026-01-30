from ...embeddings import (
    EmbeddingsModelProvider,
    EmbeddingsModelSpec,
    SentenceTransformerEmbeddingsModel
)

from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbeddingsModelProvider(EmbeddingsModelProvider):
    def can_create(self, spec: EmbeddingsModelSpec) -> bool:
        return spec.provider in {"st", "sentence_transformers", "sentence-transformers"}

    def create(self, spec: EmbeddingsModelSpec) -> Embeddings:
        if not spec.model_name:
            raise ValueError("SentenceTransformer model_name is required")

        return SentenceTransformerEmbeddingsModel(
            model_name=spec.model_name,
            cache_folder=spec.cache_folder,
            model_kwargs=spec.model_kwargs,
            encode_kwargs=spec.encode_kwargs,
            query_encode_kwargs=spec.query_encode_kwargs,
            multi_process=spec.multi_process,
            show_progress=spec.show_progress,
        )
