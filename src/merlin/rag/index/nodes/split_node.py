from langchain_text_splitters.base import TextSplitter

from merlin.rag.index.index_state import IndexState


class SplitNode:
    def __init__(self, splitter: TextSplitter):
        self.splitter = splitter

    def __call__(self, state: IndexState) -> IndexState:
        documents = state["documents"]

        splits = splitter.split_documents(documents)

        return {"splits": splits}
