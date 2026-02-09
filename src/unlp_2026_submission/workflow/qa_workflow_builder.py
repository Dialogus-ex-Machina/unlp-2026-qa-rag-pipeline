from langgraph.graph import StateGraph

from .nodes import BaseNode
from .state.qa_workflow_state import QAWorkflowState


class QAWorkflowBuilder:
    _question_answer_node: BaseNode
    _documents_retrieval_node: BaseNode
    _augmentation_node: BaseNode

    _state_graph: StateGraph

    @staticmethod
    def create() -> 'QAWorkflowBuilder':
        return QAWorkflowBuilder()

    def with_question_answering_node(
            self,
            question_answer_node: BaseNode,
    ) -> 'QAWorkflowBuilder':
        self._question_answer_node = question_answer_node
        return self

    def with_documents_retrieval_node(
            self,
            document_retrieval_node: BaseNode,
    ) -> 'QAWorkflowBuilder':
        self._documents_retrieval_node = document_retrieval_node
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
