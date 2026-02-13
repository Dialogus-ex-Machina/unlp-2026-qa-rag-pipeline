from langchain_core.vectorstores import VectorStore

from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.state import QAWorkflowState

MOST_RELEVANT_DOCUMENT_AUGMENTATION_NODE = 'most_relevant_document_augmentation_node'

class MostRelevantDocumentAugmentationNode(BaseNode):
    _vector_store: VectorStore
    _top_k: int

    def __init__(self):
        super().__init__(MOST_RELEVANT_DOCUMENT_AUGMENTATION_NODE)

    def __call__(self, state: QAWorkflowState):
        question = state['question']

        is_relevant_context_exist = bool(state.get('relevant_context', None))

        if is_relevant_context_exist:
            print('Relevant context already exists. Skipping augmentation.')
            return {}

        relevant_documents = state['relevant_documents']

        most_relevant_document = relevant_documents[0]

        if not most_relevant_document:
            print('Most relevant document not found.')
            return {
                'relevant_context': '',
                'relevant_document_id': 'UNKNOWN_ID',
                'relevant_document_page_num': -1,
            }

        relevant_context = most_relevant_document.text
        relevant_document_id = most_relevant_document.document_id
        relevant_document_page_num = most_relevant_document.page_number

        print('Retrieved doc_id:', relevant_document_id)
        print('Retrieved page_num:', relevant_document_page_num)
        print('Correct doc_id:', question['doc_id'])
        print('Correct page_num:', question['page_num'])

        return {
            'relevant_context': relevant_context,
            'relevant_document_id': relevant_document_id,
            'relevant_document_page_num': relevant_document_page_num,
        }
