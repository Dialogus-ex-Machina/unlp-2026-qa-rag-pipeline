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
            for document in document_pages:
                metadata = document.metadata or {}
                page = metadata.get("page")
                if isinstance(page, int):
                    page_number = page + 1
                    metadata["page"] = page_number
                    metadata["page_label"] = str(page_number)
                    document.metadata = metadata
            documents.extend(document_pages)

        logging.getLogger("pypdf._reader").setLevel(logging.WARNING)

        return {"documents": documents}
