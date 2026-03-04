from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from merlin.rag.index.index_state import IndexState


class HybridEmbedStoreNode:
    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = "default",
        distance: Distance = Distance.COSINE,
        batch_size: int = 64,
        dense_vector_name: str = "dense",
        sparse_vector_name: str = "sparse",
        sparse_model_name: str = "Qdrant/bm25",
        sparse_on_disk: bool = False,
        validate_collection_config: bool = True,
    ):
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.distance = distance
        self.batch_size = batch_size
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.sparse_embeddings = FastEmbedSparse(model_name=sparse_model_name)
        self.sparse_on_disk = sparse_on_disk
        self.validate_collection_config = validate_collection_config

    def __call__(self, state: IndexState) -> IndexState:
        splits: List = state.get("splits", [])
        if not splits:
            return {}

        client = QdrantClient(path=state["vector_store_path"])

        if not client.collection_exists(self.collection_name):
            dim = self._embedding_dim()
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_vector_name: VectorParams(size=dim, distance=self.distance)
                },
                sparse_vectors_config={
                    self.sparse_vector_name: SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=self.sparse_on_disk)
                    )
                },
            )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=self.dense_vector_name,
            sparse_vector_name=self.sparse_vector_name,
            validate_collection_config=self.validate_collection_config,
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
        return len(self.embeddings.embed_query("dimension_probe"))