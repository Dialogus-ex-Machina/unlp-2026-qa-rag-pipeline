from pathlib import Path
from langchain_core.documents import Document

class RelevantDocument:
    source: str
    text: str
    page_label: str
    relevance_score: float

    def __init__(
            self,
            source: str,
            text: str,
            page_label: str,
            relevance_score: float = 0.0
    ):
        self.source = source
        self.text = text
        self.page_label = page_label
        self.relevance_score = relevance_score

    @property
    def document_id(self):
        return Path(self.source).name

    @property
    def page_number(self):
        return int(self.page_label)

    @classmethod
    def from_node_with_score(cls, doc_with_score: tuple[Document, float]):
        document = doc_with_score[0]
        relevance_score = doc_with_score[1]
        metadata = document.metadata or {}
        page_label = cls._resolve_page_label(metadata)

        return RelevantDocument(
            source=metadata.get("source", "UNKNOWN_SOURCE"),
            text=document.page_content,
            page_label=str(page_label),
            relevance_score=relevance_score,
        )

    @staticmethod
    def _resolve_page_label(metadata: dict) -> str:
        page_label = metadata.get("page_label")
        if page_label is not None:
            return str(page_label)

        page = metadata.get("page", -1)
        if isinstance(page, int) and page >= 0:
            # Backward compatibility for older vector stores that persisted bare 0-based page metadata.
            return str(page + 1)

        return str(page)

    @classmethod
    def from_nodes_with_score(cls, docs_with_score: list[tuple[Document, float]]):
        return [
            RelevantDocument.from_node_with_score(doc_with_score)
            for doc_with_score in docs_with_score
        ]
