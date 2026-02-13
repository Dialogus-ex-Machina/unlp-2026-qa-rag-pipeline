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

        return RelevantDocument(
            source=document.metadata['source'],
            text=document.page_content,
            page_label=document.metadata['page_label'],
            relevance_score=relevance_score,
        )

    @classmethod
    def from_nodes_with_score(cls, docs_with_score: list[tuple[Document, float]]):
        return [
            RelevantDocument.from_node_with_score(doc_with_score)
            for doc_with_score in docs_with_score
        ]
