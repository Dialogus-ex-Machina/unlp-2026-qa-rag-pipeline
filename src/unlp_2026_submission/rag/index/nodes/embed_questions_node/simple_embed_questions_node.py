from typing import Callable, Optional

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from unlp_2026_submission.entities import Question
from unlp_2026_submission.rag.index.index_state import IndexState
from unlp_2026_submission.models.embeddings import (
    EmbeddingsModel,
    SentenceTransformerEmbeddingModel
)

class EmbeddingsModelWrapper(EmbeddingsModel):
    def __init__(self, embeddings_model: SentenceTransformerEmbeddingModel):
        self.embeddings_model = embeddings_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings_model.embed_query(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embeddings_model.embed_query(text)

class SimpleEmbedQuestionsNode:
    vector_store_client: QdrantClient
    questions: list[Question]
    embeddings_model: EmbeddingsModel
    collection_name: str
    sparse_model_name: str
    on_success: Optional[Callable[[], None]] = None
    batch_size: int

    def __init__(
            self,
            vector_store_client: QdrantClient,
            embeddings_model: SentenceTransformerEmbeddingModel,
            questions: list[Question],
            collection_name: str = "default_questions",
            on_success: Optional[Callable[[], None]] = None,
            batch_size: int = 64,
    ):
        super().__init__()

        self.questions = questions
        self.embeddings_model = EmbeddingsModelWrapper(embeddings_model)
        self.collection_name = collection_name
        self.batch_size = batch_size

        self.vector_store_client = vector_store_client
        self.on_success = on_success

    def __call__(self, state: IndexState) -> IndexState:
        if not self.vector_store_client.collection_exists(self.collection_name):
            dim = self._embedding_dim()
            self.vector_store_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        vector_store = QdrantVectorStore(
            client=self.vector_store_client,
            collection_name=self.collection_name,
            embedding=self.embeddings_model
        )

        documents = []

        for question in self.questions:
            page_content = question['question_text']
            metadata = { "id": question['question_id'] }
            document = Document(
                page_content=page_content,
                metadata=metadata
            )
            documents.append(document)

        vector_store.add_documents(
            documents=documents,
            batch_size=self.batch_size,
        )

        if self.on_success:
            self.on_success()

        return {}

    def _embedding_dim(self) -> int:
        client = getattr(self.embeddings_model, "client", None)

        if client is not None and hasattr(client, "get_sentence_embedding_dimension"):
            return int(client.get_sentence_embedding_dimension())
        else:
            return len(self.embeddings_model.embed_query("dimension_probe"))

