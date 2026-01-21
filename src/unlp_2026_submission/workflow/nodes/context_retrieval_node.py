from unlp_2026_submission.config import Config
from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.state import WorkflowState
from unlp_2026_submission.language_models import LanguageModel

CONTEXT_RETRIEVAL_NODE_NAME = 'context_retrieval_node'

class ContextRetrievalNode(BaseNode):
    def __init__(
            self,
            config: Config,
            language_model: LanguageModel
    ):
        super().__init__(
            name=CONTEXT_RETRIEVAL_NODE_NAME,
            config=config,
            language_model=language_model
        )

    def __call__(self, state: WorkflowState):
        return {
            # TODO add real retrieval
            'reference_document_id': '4e779acee13fa6e0763fb33d1c83030b8e6ea33d.pdf',
            'reference_document_page': 1,
        }
