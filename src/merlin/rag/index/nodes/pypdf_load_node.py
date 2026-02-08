from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from merlin.rag.index.index_state import IndexState


class PyPDFLoadNode:
    def __call__(self, state: IndexState) -> IndexState:
        filepaths = state["filepaths"]

        documents: List[Document] = []

        for filepath in filepaths:
            loader = PyPDFLoader(filepath)
            document_pages = loader.load()
            documents.extend(document_pages)

        return {"documents": documents}