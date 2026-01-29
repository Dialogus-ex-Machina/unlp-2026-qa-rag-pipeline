from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingsModel(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embeddings models."""

    model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", alias="model"
    )
    cache_folder: str | None = None
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    multi_progress: bool = False
    show_progress: bool = True

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._transformer = SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        preprocessed_texts = self._preprocess_text(texts)

        if self.multi_process:
            pool = self._transformer.start_multi_process_pool()
            embeddings = self._transformer.encode_document(
                sentences=preprocessed_texts,
                pool=pool,
                **self.encode_kwargs,
            )
            SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._transformer.encode_document(
                sentences=preprocessed_texts,
                show_progress_bar=self.show_progress,
                **self.encode_kwargs,
            )

        self._validate_embeddings(embeddings)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embed_kwargs = self.query_encode_kwargs or self.encode_kwargs
        preprocessed_texts = self._preprocess_text(texts)

        if self.multi_process:
            pool = self._transformer.start_multi_process_pool()
            embeddings = self._transformer.encode_document(
                sentences=preprocessed_texts,
                pool=pool,
                **embed_kwargs,
            )
            SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._transformer.encode_document(
                sentences=preprocessed_texts,
                show_progress_bar=self.show_progress,
                **embed_kwargs,
            )

        self._validate_embeddings(embeddings)
        return embeddings.tolist()[0]

    @staticmethod
    def _preprocess_text(texts: list[str]) -> list[str]:
        return [x.replace("\n", " ") for x in texts]

    @staticmethod
    def _validate_embeddings(embeddings: Any) -> None:
        if isinstance(embeddings, list):
            raise TypeError(
                "Expected embeddings to be a Tensor or a numpy array, got a list instead."
            )
