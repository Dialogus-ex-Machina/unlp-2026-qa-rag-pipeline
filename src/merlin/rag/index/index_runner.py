from itertools import pairwise

from langgraph.graph import StateGraph

from merlin.rag.index import IndexState
from merlin.rag.index.nodes import IndexNode


class IndexRunner:
    def __init__(self, nodes: list[IndexNode]):
        if not nodes:
            raise ValueError("IndexRunner requires at least one node")

        graph = StateGraph(IndexState)

        node_names: list[str] = []
        for i, node in enumerate(nodes):
            cls_name = node.__class__.__name__
            short = f"{(id(node) & 0xFFFF):04x}"
            name = f"{i:02d}-{cls_name}-{short}"
            node_names.append(name)
            graph.add_node(name, node)

        graph.set_entry_point(node_names[0])
        graph.set_finish_point(node_names[-1])

        for a, b in pairwise(node_names):
            graph.add_edge(a, b)

        self.graph = graph.compile()

    def run(self, state: IndexState):
        return self.graph.invoke(state)

