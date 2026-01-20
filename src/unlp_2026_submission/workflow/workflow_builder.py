from langgraph.graph import StateGraph

from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import LanguageModel
from unlp_2026_submission.workflow.nodes import CallLLMNode
from unlp_2026_submission.workflow.state.workflow_state import WorkflowState


class WorkflowBuilder:
    _config: Config
    _language_model: LanguageModel
    _state_graph: StateGraph

    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def create(config: Config) -> 'WorkflowBuilder':
        return WorkflowBuilder(config)

    def with_language_model(self, language_model: LanguageModel) -> 'WorkflowBuilder':
        self._language_model = language_model
        return self

    def build(self):
        self._state_graph = StateGraph(
            state_schema=WorkflowState
        )

        call_llm_node = CallLLMNode(
            config=self._config,
            language_model=self._language_model
        )

        self._state_graph.add_node(call_llm_node.name, call_llm_node)

        self._state_graph.set_entry_point(call_llm_node.name)
        self._state_graph.set_finish_point(call_llm_node.name)

        return self._state_graph.compile()
