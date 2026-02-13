from langchain_core.documents import Document
from pathlib import Path


class RelevantDocument:
    document_id: str
    text: str
    page_number: int
    score: float

    def __init__(
            self,
            document_id: str,
            text: str,
            page_number: int,
            score: float = 0.0
    ):
        self.document_id = document_id
        self.text = text
        self.page_number = page_number
        self.score = score

    @classmethod
    def from_node_with_score(cls, doc_with_score: tuple[Document, float]):
        document = doc_with_score[0]
        score = doc_with_score[1]

        return RelevantDocument(
            document_id=Path(document.metadata['source']).name,
            text=document.page_content,
            page_number=int(document.metadata['page_label']),
            score=score,
        )

    @classmethod
    def from_nodes_with_score(cls, docs_with_score: list[tuple[Document, float]]):
        return [RelevantDocument.from_node_with_score(doc_with_score) for doc_with_score in docs_with_score]
