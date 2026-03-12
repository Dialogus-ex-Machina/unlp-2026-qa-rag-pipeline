from itertools import pairwise

from langgraph.constants import END
from langgraph.graph import StateGraph

from .nodes import BaseNode
from .state import QAState


class QAWorkflowBuilder:
    _nodes: list[BaseNode] = []
    _state_graph: StateGraph

    @staticmethod
    def create() -> 'QAWorkflowBuilder':
        return QAWorkflowBuilder()

    def add_nodes(self, nodes: list[BaseNode]):
        self._nodes = nodes
        return self

    def build(self):
        self._state_graph = StateGraph(state_schema=QAState)

        node_names = self._build_nodes()
        self._state_graph.set_entry_point(node_names[0])

        return self._state_graph.compile()

    def _build_nodes(self):
        node_names: list[str] = []
        for i, node in enumerate(self._nodes):
            cls_name = node.__class__.__name__
            node_name = f"{i:02d}_{cls_name}"
            node_names.append(node_name)
            self._state_graph.add_node(node_name, node)

        for a, b in pairwise(node_names):
            self._state_graph.add_edge(a, b)

        self._state_graph.add_edge(node_names[-1], END)
        return node_names
