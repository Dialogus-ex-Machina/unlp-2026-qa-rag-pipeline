from qdrant_client import QdrantClient, models

from .embeddings_model import EmbeddingsModel

class StoredQueriesEmbeddingsModel(EmbeddingsModel):
    vector_store_client: QdrantClient
    collection_name: str

    def __init__(
            self,
            vector_store_client: QdrantClient,
            collection_name: str = "default_questions",
    ):
        self.vector_store_client = vector_store_client
        self.collection_name = collection_name


    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = self.vector_store_client.query_points(
            collection_name=self.collection_name,
            with_vectors=True,
            limit=len(texts),
        ).points

        embeddings = [result.vector for result in results]

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        results = self.vector_store_client.query_points(
            collection_name=self.collection_name,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="page_content",
                        match=models.MatchValue(value=text),
                    ),
                ]
            ),
            with_vectors=True,
        ).points

        return results[0].vector