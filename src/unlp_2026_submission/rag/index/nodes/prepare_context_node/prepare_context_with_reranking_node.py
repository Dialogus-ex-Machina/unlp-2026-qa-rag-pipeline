from typing import Callable, Optional, Dict, Any
import gc
import random
import torch

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from unlp_2026_submission.entities import Question, RelevantDocument
from unlp_2026_submission.models.reranker_models import BGEV2RerankerModel
from unlp_2026_submission.rag.index.index_state import IndexState
from unlp_2026_submission.models.embeddings import EmbeddingsModel, StoredQueriesEmbeddingsModel


class PrepareContextWithRerankingNode:
    vector_store_client: QdrantClient
    embeddings_model: EmbeddingsModel
    documents_collection_name: str
    questions_collection_name: str
    top_k: int
    context_size_1: int
    context_size_2: int
    context_size_1_probability: float
    batch_size: int
    reranker_kwargs: Dict[str, Any]

    def __init__(
            self,
            vector_store_client: QdrantClient,
            reranker_kwargs: Dict[str, Any],
            documents_collection_name: str = "default",
            questions_collection_name: str = "default_questions",
            top_k: int = 10,
            context_size_1: int = 3,
            context_size_2: int = 2,
            context_size_1_probability: float = 0.70,
            on_success: Optional[Callable[[], None]] = None,
    ):
        super().__init__()

        self.vector_store_client = vector_store_client
        self.on_success = on_success

        self.embeddings_model = StoredQueriesEmbeddingsModel(
            vector_store_client=vector_store_client,
            collection_name=questions_collection_name
        )
        self.top_k = top_k

        self.context_size_1 = context_size_1
        self.context_size_2 = context_size_2
        self.context_size_1_probability = context_size_1_probability

        self.documents_collection_name = documents_collection_name
        self.reranker_kwargs = reranker_kwargs

    def __call__(self, state: IndexState) -> IndexState:
        questions: list[Question] = state['questions']

        vector_store = QdrantVectorStore(
            client=self.vector_store_client,
            collection_name=self.documents_collection_name,
            embedding=self.embeddings_model
        )

        reranker_model = BGEV2RerankerModel(
            **self.reranker_kwargs
        )

        for question in questions:
            docs_with_score = vector_store.similarity_search_with_score(
                question['question_text'],
                k=self.top_k
            )

            relevant_documents = RelevantDocument.from_nodes_with_score(docs_with_score)

            reranked_relevant_documents = reranker_model.rerank(
                query=question['question_text'],
                documents=relevant_documents,
            )

            relevant_context = self._create_relevant_context(reranked_relevant_documents)

            relevant_document_id = reranked_relevant_documents[0].document_id
            relevant_document_page_num = reranked_relevant_documents[0].page_number

            question['doc_text'] = relevant_context
            question['doc_id'] = relevant_document_id
            question['page_num'] = relevant_document_page_num

        if self.on_success:
            self.on_success()

        del reranker_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            'questions': questions,
        }

    def _create_relevant_context(self, relevant_documents: list[RelevantDocument]) -> str:
        bucket = random.random()

        current_context_size = (
            self.context_size_1
            if bucket < self.context_size_1_probability
            else self.context_size_2
        )

        k = min(current_context_size, len(relevant_documents))
        docs = relevant_documents[:k]

        separator = "\n\n"
        relevant_context = separator.join(doc.text for doc in docs)

        return relevant_context