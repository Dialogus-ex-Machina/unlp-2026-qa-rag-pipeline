from langchain_text_splitters.base import TextSplitter

from merlin.rag.index.index_state import IndexState


class SplitNode:
    def __init__(self, splitter: TextSplitter = None):
        self.splitter = splitter

    def __call__(self, state: IndexState) -> IndexState:
        print("Splitting...")
        
        documents = state["documents"]

        if self.splitter != None:
            return {"splits": self.splitter.split_documents(documents)}
        else:
            return {"splits": documents}
