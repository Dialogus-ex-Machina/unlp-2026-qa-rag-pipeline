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

        return {"documents": documents}
