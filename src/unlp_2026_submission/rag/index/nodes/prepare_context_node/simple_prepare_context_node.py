from typing import Callable, Optional

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from unlp_2026_submission.entities import Question, RelevantDocument
from unlp_2026_submission.rag.index.index_state import IndexState
from unlp_2026_submission.models.embeddings import EmbeddingsModel, StoredQueriesEmbeddingsModel


class SimplePrepareContextNode:
    vector_store_client: QdrantClient
    embeddings_model: EmbeddingsModel
    collection_name: str
    top_k: int
    context_window_size: int
    batch_size: int

    def __init__(
            self,
            vector_store_client: QdrantClient,
            collection_name: str = "default",
            top_k: int = 10,
            context_window_size: int = 3,
            on_success: Optional[Callable[[], None]] = None,
    ):
        super().__init__()

        self.vector_store_client = vector_store_client
        self.on_success = on_success

        self.embeddings_model = StoredQueriesEmbeddingsModel(
            vector_store_client
        )
        self.top_k = top_k
        self.context_window_size = context_window_size
        self.collection_name = collection_name

    def __call__(self, state: IndexState) -> IndexState:
        questions: list[Question] = state['questions']

        vector_store = QdrantVectorStore(
            client=self.vector_store_client,
            collection_name=self.collection_name,
            embedding=self.embeddings_model
        )

        for question in questions:
            docs_with_score = vector_store.similarity_search_with_score(
                question['question_text'],
                k=self.top_k
            )

            relevant_documents = RelevantDocument.from_nodes_with_score(docs_with_score)

            relevant_context = self._create_relevant_context(relevant_documents)

            relevant_document_id = relevant_documents[0].document_id
            relevant_document_page_num = relevant_documents[0].page_number

            question['doc_text'] = relevant_context
            question['doc_id'] = relevant_document_id
            question['page_num'] = relevant_document_page_num

        if self.on_success:
            self.on_success()

        return {
            'questions': questions,
        }

    def _create_relevant_context(self, relevant_documents: list[RelevantDocument]) -> str:
        k = min(self.context_window_size, len(relevant_documents))
        docs = relevant_documents[:k]
        top_k_relevant = docs

        separator = "\n\n"
        relevant_context = separator.join(doc.text for doc in top_k_relevant)

        return relevant_context

