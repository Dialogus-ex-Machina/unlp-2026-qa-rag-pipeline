from typing import TypedDict, Required, List

from langchain_core.documents import Document

from unlp_2026_submission.entities import Question

class IndexState(TypedDict, total=False):
    filepaths: List[str]
    documents: List[Document]
    splits: List[Document]
    vector_store_path: Required[str]
    questions: List[Question]

