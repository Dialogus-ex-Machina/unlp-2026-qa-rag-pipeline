from typing import Literal

from unlp_2026_submission.entities import RelevantDocument
from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.state import QAWorkflowState

class TopKRelevantDocumentAugmentation(BaseNode):
    _top_k: int
    _should_reorder: bool

    def __init__(
            self,
            top_k: Literal[1, 2, 3, 4, 5] = 3,
            should_reorder: bool = False,
    ):
        self._top_k = top_k
        self._should_reorder = should_reorder

    def __call__(self, state: QAWorkflowState):
        question = state['question']

        is_relevant_context_exist = bool(state.get('relevant_context', None))

        relevant_documents = state['relevant_documents']

        if is_relevant_context_exist:
            print('Relevant context already exists. Skipping augmentation.')
            return {}


        if not relevant_documents:
            print('Most relevant document not found.')
            return {
                'relevant_context': '',
                'relevant_document_id': 'UNKNOWN_ID',
                'relevant_document_page_num': -1,
            }

        relevant_context = self._create_relevant_context(relevant_documents)

        relevant_document_id = relevant_documents[0].document_id
        relevant_document_page_num = relevant_documents[0].page_number

        print('Retrieved doc_id:', relevant_document_id)
        print('Retrieved page_num:', relevant_document_page_num)
        print('Correct doc_id:', question['doc_id'])
        print('Correct page_num:', question['page_num'])

        return {
            'relevant_context': relevant_context,
            'relevant_document_id': relevant_document_id,
            'relevant_document_page_num': relevant_document_page_num,
        }

    def _create_relevant_context(self, relevant_documents: list[RelevantDocument]) -> str:
        k = min(self._top_k, len(relevant_documents))
        docs = relevant_documents[:k]

        if self._should_reorder:
            top_k_relevant = self._reorder_docs(docs)
        else:
            top_k_relevant = docs

        separator = "\n\n"
        relevant_context = separator.join(doc.text for doc in top_k_relevant)

        return relevant_context

    def _reorder_docs(self, docs: list[RelevantDocument]) -> list[RelevantDocument]:
        n = len(docs)
        # evens: 0,2,4,...
        evens = list(range(0, n, 2))
        # odds: 1,3,5,... then reverse -> ...,3,1
        odds = list(range(1, n, 2))[::-1]
        order = evens + odds
        reordered = [docs[i] for i in order]

        return reordered
