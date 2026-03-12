from typing import List

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain_core.embeddings import Embeddings

from unlp_2026_submission.rag.index.index_state import IndexState


class EmbedStoreNode:
    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "default",
        distance: Distance = Distance.COSINE,
        batch_size: int = 64,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.distance = distance
        self.batch_size = batch_size

    def __call__(self, state: IndexState) -> IndexState:
        splits: List = state.get("splits", [])
        if not splits:
            return {}

        qdrant_client = QdrantClient(path=state["vector_store_path"])

        if not qdrant_client.collection_exists(self.collection_name):
            dim = self._embedding_dim()
            qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=self.distance),
            )

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        vector_store.add_documents(
            documents=splits,
            batch_size=self.batch_size,
        )

        return {}

    def _embedding_dim(self) -> int:
        client = getattr(self.embeddings, "client", None)

        if client is not None and hasattr(client, "get_sentence_embedding_dimension"):
            return int(client.get_sentence_embedding_dimension())
        else:
            return len(self.embeddings.embed_query("dimension_probe"))
