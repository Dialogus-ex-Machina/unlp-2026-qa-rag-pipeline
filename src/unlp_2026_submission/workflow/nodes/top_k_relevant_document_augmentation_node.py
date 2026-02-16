from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.state import QAWorkflowState

class TopKRelevantDocumentAugmentation(BaseNode):
    _top_k: int

    def __init__(self, top_k: int = 3):
        self._top_k = top_k

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

        top_k_relevant = relevant_documents[0:self._top_k]

        # separator = "\n---------------------\n"
        separator = "\n\n"
        relevant_context = separator.join(
            page.text for page in top_k_relevant
        )

        relevant_document_id = top_k_relevant[0].document_id
        relevant_document_page_num = top_k_relevant[0].page_number

        print('Retrieved doc_id:', relevant_document_id)
        print('Retrieved page_num:', relevant_document_page_num)
        print('Correct doc_id:', question['doc_id'])
        print('Correct page_num:', question['page_num'])

        return {
            'relevant_context': relevant_context,
            'relevant_document_id': relevant_document_id,
            'relevant_document_page_num': relevant_document_page_num,
        }
