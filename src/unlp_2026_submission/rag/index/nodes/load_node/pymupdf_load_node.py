from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from unlp_2026_submission.rag.index.index_state import IndexState


class PyMuPDFLoadNode:
    def __call__(self, state: IndexState) -> IndexState:
        filepaths = state["filepaths"]

        documents: List[Document] = []

        for filepath in filepaths:
            loader = PyMuPDFLoader(filepath, mode="page")
            documents.extend(loader.load())

        return {"documents": documents}
