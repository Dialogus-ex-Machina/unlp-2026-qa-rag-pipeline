from langchain_text_splitters import RecursiveCharacterTextSplitter

from merlin.rag.index.index_state import IndexState


class SplitNode:
    def __call__(self, state: IndexState) -> IndexState:
        documents = state["documents"]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            add_start_index=True,
        )

        splits = text_splitter.split_documents(documents)

        return {"splits": splits}
