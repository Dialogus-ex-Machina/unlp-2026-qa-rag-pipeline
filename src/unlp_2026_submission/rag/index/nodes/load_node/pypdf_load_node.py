from typing import List
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from unlp_2026_submission.rag.index.index_state import IndexState


class PyPDFLoadNode:
    def __call__(self, state: IndexState) -> IndexState:
        logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

        filepaths = state["filepaths"]

        documents: List[Document] = []

        for filepath in filepaths:
            loader = PyPDFLoader(filepath)
            document_pages = loader.load()
            documents.extend(document_pages)

        logging.getLogger("pypdf._reader").setLevel(logging.WARNING)

        return {"documents": documents}
