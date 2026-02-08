import os
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

from unlp_2026_submission.config import KnowledgeBaseConfig
from unlp_2026_submission.embeddings import EmbeddingsModel
from unlp_2026_submission.entities import DocumentPage


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class LangchainKnowledgeBase:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        embeddings_model: EmbeddingsModel,
        config: KnowledgeBaseConfig,
    ):
        self._vector_store = vector_store
        self._embeddings_model = embeddings_model
        self._config = config
        self._doc_id_key = os.getenv("KB_DOC_ID_KEY")
        self._page_key = os.getenv("KB_PAGE_KEY")
        self._page_offset = _env_int("KB_PAGE_NUMBER_OFFSET", 0)
        self._doc_id_basename = _env_bool("KB_DOC_ID_BASENAME", True)
        self._similarity_cutoff = _env_float("KB_SIMILARITY_CUTOFF", 0.5)
        self._score_direction = os.getenv("KB_SCORE_DIRECTION", "higher").strip().lower()

    @property
    def vector_store(self):
        return self._vector_store

    def _get_metadata_value(self, metadata: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key and key in metadata and metadata[key] is not None:
                return metadata[key]
        return None

    def _resolve_document_id(self, metadata: dict[str, Any]) -> str | None:
        keys = [self._doc_id_key] if self._doc_id_key else [
            "document_id",
            "doc_id",
            "file_name",
            "source",
        ]
        value = self._get_metadata_value(metadata, keys)
        if value is None:
            return None
        value = str(value)
        if self._doc_id_basename and ("/" in value or "\\" in value):
            return os.path.basename(value)
        return value

    def _resolve_page_number(self, metadata: dict[str, Any]) -> int | None:
        keys = [self._page_key] if self._page_key else [
            "page_label",
            "page",
            "page_number",
            "page_num",
        ]
        value = self._get_metadata_value(metadata, keys)
        if value is None:
            return None
        try:
            page_num = int(value)
        except (TypeError, ValueError):
            try:
                page_num = int(float(value))
            except (TypeError, ValueError):
                return None
        return page_num + self._page_offset

    def _document_to_page(self, doc: Document) -> DocumentPage | None:
        metadata = doc.metadata or {}
        document_id = self._resolve_document_id(metadata)
        page_number = self._resolve_page_number(metadata)
        if document_id is None or page_number is None:
            return None
        return DocumentPage(
            document_id=document_id,
            text=doc.page_content,
            page_number=page_number,
        )

    def retrieve_page(
        self,
        search_query: str,
        filters: Filter | None = None,
        hybrid_top_k: int = 2,
        similarity_top_k: int = 2,
        sparse_top_k: int = 2,
    ) -> DocumentPage | None:
        del hybrid_top_k, sparse_top_k

        results = self._vector_store.similarity_search_with_score(
            query=search_query,
            k=similarity_top_k,
            filter=filters,
        )

        if self._score_direction == "lower":
            filtered = [
                (doc, score)
                for doc, score in results
                if score <= self._similarity_cutoff
            ]
        else:
            filtered = [
                (doc, score)
                for doc, score in results
                if score >= self._similarity_cutoff
            ]

        if not filtered:
            return None

        doc, _score = filtered[0]
        return self._document_to_page(doc)

    def retrieve(
        self,
        search_query: str,
        filters: Filter | None = None,
        hybrid_top_k: int = 2,
        similarity_top_k: int = 2,
        sparse_top_k: int = 2,
    ) -> list[Document]:
        del hybrid_top_k, sparse_top_k
        return self._vector_store.similarity_search(
            query=search_query,
            k=similarity_top_k,
            filter=filters,
        )

    def query(
        self,
        search_query: str,
        filters: Filter | None = None,
        hybrid_top_k: int = 2,
        similarity_top_k: int = 2,
        sparse_top_k: int = 2,
    ) -> list[Document]:
        return self.retrieve(
            search_query=search_query,
            filters=filters,
            hybrid_top_k=hybrid_top_k,
            similarity_top_k=similarity_top_k,
            sparse_top_k=sparse_top_k,
        )

    @classmethod
    def load(
        cls,
        llama_index_language_model: Any,
        embeddings_model: EmbeddingsModel,
        config: KnowledgeBaseConfig,
        should_persist: bool = True,
    ):
        del llama_index_language_model

        if should_persist and not os.path.exists(config.vector_store_path):
            raise FileNotFoundError(
                f"Vector store path does not exist: {config.vector_store_path}"
            )

        if should_persist:
            qdrant_client = QdrantClient(path=config.vector_store_path)
        else:
            qdrant_client = QdrantClient(location=":memory:")

        if not qdrant_client.collection_exists(config.collection_name):
            raise ValueError(
                f"Qdrant collection does not exist: {config.collection_name}"
            )

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=config.collection_name,
            embedding=embeddings_model,
        )

        return cls(
            vector_store=vector_store,
            embeddings_model=embeddings_model,
            config=config,
        )
