from merlin.rag.index.index_state import IndexState


class ProxySplitNode:
    def __call__(self, state: IndexState) -> IndexState:
        documents = state["documents"]

        return {"splits": documents}
