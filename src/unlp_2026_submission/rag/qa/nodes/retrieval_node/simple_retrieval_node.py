from langchain_core.vectorstores import VectorStore

from unlp_2026_submission.entities import RelevantDocument
from unlp_2026_submission.rag.qa.nodes.base_node import BaseNode
from unlp_2026_submission.rag.qa.state import QAWorkflowState

class SimpleRetrievalNode(BaseNode):
    _vector_store: VectorStore
    _top_k: int

    def __init__(
            self,
            vector_store: VectorStore,
            top_k: int = 10,
    ):
        super().__init__()
        self._vector_store = vector_store
        self._top_k = top_k

    def __call__(self, state: QAWorkflowState):
        question = state['question']

        is_relevant_context_exist = bool(state.get('relevant_context', None))

        if is_relevant_context_exist:
            print('Relevant context already exists. Skipping context retrieval.')
            return {}

        docs_with_score = self._vector_store.similarity_search_with_score(
            question['question_text'],
            k=self._top_k
        )

        relevant_documents = RelevantDocument.from_nodes_with_score(docs_with_score)

        return {
            'relevant_documents': relevant_documents,
        }
