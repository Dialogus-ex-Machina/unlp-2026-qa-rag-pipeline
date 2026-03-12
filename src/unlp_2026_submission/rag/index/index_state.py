from typing import TypedDict, Required, List

from langchain_core.documents import Document


class IndexState(TypedDict, total=False):
    filepaths: List[str]
    documents: List[Document]
    splits: List[Document]
    vector_store_path: Required[str]
