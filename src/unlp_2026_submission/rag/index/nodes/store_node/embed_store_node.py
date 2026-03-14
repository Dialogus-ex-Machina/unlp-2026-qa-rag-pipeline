from typing import List, Dict, Any
import gc

import torch
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from unlp_2026_submission.models.embeddings import SentenceTransformerEmbeddingModel, EmbeddingsModel
from unlp_2026_submission.rag.index.index_state import IndexState


class EmbedStoreNode:
    vector_store_client: QdrantClient
    embeddings_model_kwargs: Dict[str, Any]

    def __init__(
        self,
        vector_store_client: QdrantClient,
        embeddings_model_kwargs: Dict[str, Any],
        collection_name: str = "default",
        distance: Distance = Distance.COSINE,
        batch_size: int = 64,
    ):
        self.vector_store_client = vector_store_client
        self.embeddings_model_kwargs = embeddings_model_kwargs
        self.collection_name = collection_name
        self.distance = distance
        self.batch_size = batch_size

    def __call__(self, state: IndexState) -> IndexState:
        splits: List = state.get("splits", [])
        if not splits:
            return {}

        embeddings_model = SentenceTransformerEmbeddingModel(
            **self.embeddings_model_kwargs
        )

        if not self.vector_store_client.collection_exists(self.collection_name):
            dim = self._embedding_dim(embeddings_model=embeddings_model)
            self.vector_store_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=self.distance),
            )

        vector_store = QdrantVectorStore(
            client=self.vector_store_client,
            collection_name=self.collection_name,
            embedding=embeddings_model,
        )

        vector_store.add_documents(
            documents=splits,
            batch_size=self.batch_size,
        )

        del embeddings_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {}

    def _embedding_dim(self, embeddings_model: EmbeddingsModel) -> int:
        client = getattr(embeddings_model, "client", None)

        if client is not None and hasattr(client, "get_sentence_embedding_dimension"):
            return int(client.get_sentence_embedding_dimension())
        else:
            return len(embeddings_model.embed_query("dimension_probe"))
