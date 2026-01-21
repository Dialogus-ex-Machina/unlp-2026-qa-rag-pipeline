from langgraph.graph import StateGraph

from unlp_2026_submission.config import Config
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.language_models import LanguageModel
from unlp_2026_submission.workflow.nodes import QuestionAnswerNode, ContextRetrievalNode
from unlp_2026_submission.workflow.state.workflow_state import WorkflowState


class WorkflowBuilder:
    _config: Config
    _language_model: LanguageModel
    _knowledge_base: KnowledgeBase
    _state_graph: StateGraph

    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def create(config: Config) -> 'WorkflowBuilder':
        return WorkflowBuilder(config)

    def with_language_model(self, language_model: LanguageModel) -> 'WorkflowBuilder':
        self._language_model = language_model
        return self

    def with_knowledge_base(self, knowledge_base: KnowledgeBase) -> 'WorkflowBuilder':
        self._knowledge_base = knowledge_base
        return self

    def build(self):
        self._state_graph = StateGraph(
            state_schema=WorkflowState
        )

        context_retrieval_node = ContextRetrievalNode(
            config=self._config,
            language_model=self._language_model,
            knowledge_base=self._knowledge_base
        )
        question_answer_node = QuestionAnswerNode(
            config=self._config,
            language_model=self._language_model,
            knowledge_base=self._knowledge_base
        )

        self._state_graph.add_node(question_answer_node.name, question_answer_node)
        self._state_graph.add_node(context_retrieval_node.name, context_retrieval_node)

        self._state_graph.set_entry_point(context_retrieval_node.name)

        self._state_graph.add_edge(context_retrieval_node.name, question_answer_node.name)

        self._state_graph.set_finish_point(question_answer_node.name)

        return self._state_graph.compile()
