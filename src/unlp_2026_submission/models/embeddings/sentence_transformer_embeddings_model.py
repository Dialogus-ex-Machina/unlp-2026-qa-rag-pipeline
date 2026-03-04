from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddingModel(BaseModel, Embeddings):
    """HuggingFace sentence_transformers embedding models."""

    model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2", alias="model"
    )
    """Model name to use."""
    cache_folder: str | None = None
    """Path to store models.
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the Sentence Transformer model, such as `device`,
    `prompts`, `default_prompt_name`, `revision`, `trust_remote_code`, or `token`.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer"""
    encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the documents of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    query_encode_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method for the query of
    the Sentence Transformer model, such as `prompt_name`, `prompt`, `batch_size`,
    `precision`, `normalize_embeddings`, and more.
    See also the Sentence Transformer documentation: https://sbert.net/docs/package_reference/SentenceTransformer.html#sentence_transformers.SentenceTransformer.encode"""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    show_progress: bool = True
    """Whether to show a progress bar."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        populate_by_name=True,
    )

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

        self._transformer = SentenceTransformer(
            self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
        )

    @staticmethod
    def create(
            model_name: str,
            cache_dir: str | None = None,
    ):
        return SentenceTransformerEmbeddingModel(
            model_name=model_name,
            cache_folder=cache_dir,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.

        """
        preprocessed_texts = self._preprocess_text(texts)

        if self.multi_process:
            pool = self._transformer.start_multi_process_pool()
            embeddings = self._transformer.encode_document(
                sentences=preprocessed_texts,
                pool=pool,
                **self.encode_kwargs
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
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.

        """
        embed_kwargs = (
            self.query_encode_kwargs
            if len(self.query_encode_kwargs) > 0
            else self.encode_kwargs
        )

        preprocessed_texts = self._preprocess_text([text])

        if self.multi_process:
            pool = self._transformer.start_multi_process_pool()
            embeddings = self._transformer.encode_query(
                sentences=preprocessed_texts,
                pool=pool,
                **embed_kwargs
            )
            SentenceTransformer.stop_multi_process_pool(pool)
        else:
            embeddings = self._transformer.encode_query(
                sentences=preprocessed_texts,
                show_progress_bar=self.show_progress,
                **embed_kwargs,
            )

        self._validate_embeddings(embeddings)

        return embeddings.tolist()[0]

    def _preprocess_text(self, texts: list[str]) -> list[str]:
        return [x.replace("\n", " ") for x in texts]

    def _validate_embeddings(self, embeddings: list[Any]) -> None:
        if isinstance(embeddings, list):
            msg = (
                "Expected embeddings to be a Tensor or a numpy array, "
                "got a list instead."
            )
            raise TypeError(msg)
