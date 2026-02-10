from itertools import pairwise

from langgraph.constants import END
from langgraph.graph import StateGraph

from unlp_2026_submission.entities import QuestionDomain

from .nodes import BaseNode
from .state.qa_workflow_state import QAWorkflowState


class QAWorkflowBuilder:
    _domain_routing_node: BaseNode
    _medicine_domain_nodes: list[BaseNode] = []
    _sport_domain_nodes: list[BaseNode] = []
    _other_domain_nodes: list[BaseNode] = []

    _state_graph: StateGraph

    @staticmethod
    def create() -> 'QAWorkflowBuilder':
        return QAWorkflowBuilder()

    def add_domain_routing_node(self, domain_routing_node: BaseNode):
        self._domain_routing_node = domain_routing_node
        return self

    def add_medicine_domain_nodes(self, medicine_domain_nodes: list[BaseNode]):
        self._medicine_domain_nodes = medicine_domain_nodes
        return self

    def add_sport_domain_nodes(self, sport_domain_nodes: list[BaseNode]):
        self._sport_domain_nodes = sport_domain_nodes
        return self

    def add_other_domain_nodes(self, other_domain_nodes: list[BaseNode]):
        self._other_domain_nodes = other_domain_nodes
        return self

    def build(self):
        domain_routing_node_name = 'domain_routing'
        self._state_graph = StateGraph(state_schema=QAWorkflowState)

        self._state_graph.add_node(domain_routing_node_name, self._domain_routing_node)
        self._state_graph.set_entry_point(domain_routing_node_name)

        medicine_node_names = self._build_domain_nodes(
            'medicine',
            self._medicine_domain_nodes
        )
        sport_node_names = self._build_domain_nodes(
            'sport',
            self._sport_domain_nodes
        )
        other_node_names = self._build_domain_nodes(
            'other',
            self._other_domain_nodes
        )

        self._state_graph.add_conditional_edges(
            domain_routing_node_name,
            lambda state: state["predicted_domain"],
            {
                "medicine": medicine_node_names[0],
                "sport": sport_node_names[0],
                "other": other_node_names[0],
            }
        )

        return self._state_graph.compile()

    def _build_domain_nodes(
            self,
            domain_name: QuestionDomain,
            nodes: list[BaseNode],
    ):
        node_names: list[str] = []
        for i, node in enumerate(nodes):
            cls_name = node.__class__.__name__
            node_name = f"{domain_name}_{i:02d}_{cls_name}"
            node_names.append(node_name)
            self._state_graph.add_node(node_name, node)

        for a, b in pairwise(node_names):
            self._state_graph.add_edge(a, b)

        self._state_graph.add_edge(node_names[-1], END)
        return node_names
