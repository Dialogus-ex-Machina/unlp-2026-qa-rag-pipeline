from unlp_2026_submission.config import Config
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.prompts import PromptsFactory
from unlp_2026_submission.workflow.state import WorkflowState
from unlp_2026_submission.language_models import LanguageModel

CONTEXT_RETRIEVAL_NODE_NAME = 'context_retrieval_node'

class ContextRetrievalNode(BaseNode):
    def __init__(
            self,
            config: Config,
            language_model: LanguageModel,
            knowledge_base: KnowledgeBase,
            prompts_factory: PromptsFactory,
    ):
        super().__init__(
            name=CONTEXT_RETRIEVAL_NODE_NAME,
            config=config,
            language_model=language_model,
            knowledge_base=knowledge_base,
            prompts_factory=prompts_factory,
        )

    def __call__(self, state: WorkflowState):
        question = state['question']

        is_ref_page_exist = bool(state.get('reference_document_page', None))

        if is_ref_page_exist:
            print('Reference page already exists. Skipping context retrieval.')
            return {}

        page = self.knowledge_base.retrieve_page(
            search_query=question['question_text']
        )

        if not page:
            # TODO handle this
            return {
                'reference_document_page': None,
                'reference_document_id': '',
                'reference_document_page_num': -1,
            }

        print('Retrieved doc_id:', page.document_id)
        print('Retrieved page_num:', page.page_number)
        print('Correct doc_id:', question['doc_id'])
        print('Correct page_num:', question['page_num'])

        return {
            'reference_document_page': page,
            'reference_document_id': page.document_id,
            'reference_document_page_num': page.page_number,
        }
