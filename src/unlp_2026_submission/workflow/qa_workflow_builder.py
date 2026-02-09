from langchain_core.vectorstores import VectorStore
from langgraph.graph import StateGraph

from unlp_2026_submission.language_models import LanguageModel
from .nodes import QuestionAnswerNode, BaseNode
from .nodes.documents_retrieval_node import DocumentsRetrievalNode
from .prompts.base_qa_prompt import BaseQAPrompt
from .state.qa_workflow_state import QAWorkflowState


class QAWorkflowBuilder:
    _question_answer_node: QuestionAnswerNode
    _documents_retrieval_node: DocumentsRetrievalNode
    _augmentation_node: BaseNode

    _state_graph: StateGraph

    @staticmethod
    def create() -> 'QAWorkflowBuilder':
        return QAWorkflowBuilder()

    def with_question_answering_node(
            self,
            prompt: BaseQAPrompt,
            language_model: LanguageModel,
    ) -> 'QAWorkflowBuilder':
        self._question_answer_node = QuestionAnswerNode(
            language_model=language_model,
            prompt=prompt,
        )
        return self

    def with_documents_retrieval_node(
            self,
            vector_store: VectorStore,
            top_k: int | None = None,
    ) -> 'QAWorkflowBuilder':
        self._documents_retrieval_node = DocumentsRetrievalNode(
            vector_store=vector_store,
            top_k=top_k
        )
        return self

    def with_augmentation_node(self, augmentation_node: BaseNode):
        self._augmentation_node = augmentation_node
        return self

    def build(self):
        self._state_graph = StateGraph(state_schema=QAWorkflowState)

        self._state_graph.add_node(self._documents_retrieval_node.name, self._documents_retrieval_node)
        self._state_graph.add_node(self._augmentation_node.name, self._augmentation_node)
        self._state_graph.add_node(self._question_answer_node.name, self._question_answer_node)

        self._state_graph.set_entry_point(self._documents_retrieval_node.name)

        self._state_graph.add_edge(self._documents_retrieval_node.name, self._augmentation_node.name)
        self._state_graph.add_edge(self._augmentation_node.name, self._question_answer_node.name)

        self._state_graph.set_finish_point(self._question_answer_node.name)

        return self._state_graph.compile()
