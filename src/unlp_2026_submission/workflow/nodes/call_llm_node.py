from unlp_2026_submission.config import Config
from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.state import WorkflowState
from unlp_2026_submission.language_models import LanguageModel

CALL_LLM_NODE_NAME = 'call_llm'

class CallLLMNode(BaseNode):
    def __init__(
            self,
            config: Config,
            language_model: LanguageModel
    ):
        super().__init__(
            name=CALL_LLM_NODE_NAME,
            config=config,
            language_model=language_model
        )

    async def __call__(self, state: WorkflowState):
        query = state['query']

        response_message = await self.language_model.ainvoke(query)

        return {
            'messages': response_message
        }
