from unlp_2026_submission.rag.qa.nodes.base_node import BaseNode
from unlp_2026_submission.rag.qa.state import QAState

class MostRelevantDocsContextCreationNode(BaseNode):
    def __call__(self, state: QAState):
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

        most_relevant_document = relevant_documents[0]

        relevant_context = most_relevant_document.text
        relevant_document_id = most_relevant_document.document_id
        relevant_document_page_num = most_relevant_document.page_number

        print('Retrieved doc_id:', relevant_document_id)
        print('Retrieved page_num:', relevant_document_page_num)

        correct_doc_id = question.get('doc_id')
        if correct_doc_id is not None:
            print('Correct doc_id:', correct_doc_id)

        correct_page_num = question.get('page_num')
        if correct_page_num is not None:
            print('Correct page_num:', correct_page_num)

        return {
            'relevant_context': relevant_context,
            'relevant_document_id': relevant_document_id,
            'relevant_document_page_num': relevant_document_page_num,
        }
