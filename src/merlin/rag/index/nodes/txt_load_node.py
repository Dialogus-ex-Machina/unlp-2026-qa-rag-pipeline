from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from merlin.rag.index.index_state import IndexState


class TxtLoadNode:
    def __call__(self, state: IndexState) -> IndexState:
        filepaths = state["filepaths"]

        documents: List[Document] = []

        for filepath in filepaths:
            loader = TextLoader(filepath)
            document_pages = loader.load()
            documents.extend(document_pages)

        return {"documents": documents}
