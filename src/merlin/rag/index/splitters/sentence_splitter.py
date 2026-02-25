"""
LangChain BaseDocumentTransformer wrapper around llama_index SentenceSplitter.
Splits text by sentences (with chunk_size/chunk_overlap in token space) and
adds start_index and end_index to each chunk's metadata.
"""
import copy
import re
from typing import Any, List, Sequence, Tuple

from langchain_core.documents import BaseDocumentTransformer, Document
from llama_index.core.node_parser import SentenceSplitter as LlamaSentenceSplitter

# - "1. " (арабські)
# - "IV. " (латинські римські)
# - "І. " (кирилична "І", часто в укр. документах)
LIST_MARKER_RE = re.compile(r"(?m)^(\s*)(\d+|[IVXLCDM]+|[І]+)\.\s+")
DOT_PLACEHOLDER = "∯"


class SentenceSplitter(BaseDocumentTransformer):
    """
    Splits documents using llama_index SentenceSplitter (sentence-aware, token-based
    chunk_size/chunk_overlap). No semantic clustering. Each output document has
    metadata start_index and end_index (character offsets in the original text).
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
    ):
        self.splitter = LlamaSentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        transformed_documents = []
        for document in documents:
            split_documents = self._split_document(document)
            transformed_documents.extend(split_documents)
        return transformed_documents

    def _split_document(self, document: Document) -> List[Document]:
        split_documents = []
        chunks = self._split_text(document.page_content)
        for chunk_text, start, end in chunks:
            doc_metadata = copy.deepcopy(document.metadata)
            doc_metadata["start_index"] = start
            doc_metadata["end_index"] = end
            split_documents.append(
                Document(page_content=chunk_text, metadata=doc_metadata)
            )
        return split_documents

    def _split_text(self, text: str) -> List[Tuple[str, int, int]]:
        preprocessed_text = self._preprocess_text(text)
        chunks = self.splitter.split_text(preprocessed_text)
        chunks_unmasked = self._postprocess_chunks(chunks)
        spans = self._find_ordered_spans(preprocessed_text, chunks)
        out: List[Tuple[str, int, int]] = []
        for i, (start, end) in enumerate(spans):
            chunk_text = chunks_unmasked[i]
            out.append((chunk_text, start, end))
        return out

    def _preprocess_text(self, text: str) -> str:
        return self._mask_list_markers(text)

    def _mask_list_markers(self, text: str) -> str:
        return LIST_MARKER_RE.sub(
            lambda m: f"{m.group(1)}{m.group(2)}{DOT_PLACEHOLDER} ", text
        )

    def _postprocess_chunks(self, chunks: List[str]) -> List[str]:
        return [self._unmask_list_markers(s) for s in chunks]

    def _unmask_list_markers(self, text: str) -> str:
        return re.sub(
            rf"(\b\d+|\b[IVXLCDM]+|\b[І]+){re.escape(DOT_PLACEHOLDER)}\s",
            r"\1. ",
            text,
        )

    def _find_ordered_spans(
        self,
        full_text: str,
        parts: List[str],
    ) -> List[Tuple[int, int]]:
        """
        Finds each part in full_text in order, starting each search after the
        previous match. Returns [(start, end_exclusive), ...].
        """
        spans: List[Tuple[int, int]] = []
        cursor = 0
        for p in parts:
            if not p:
                spans.append((cursor, cursor))
                continue
            idx = full_text.find(p, cursor)
            if idx == -1:
                raise ValueError(
                    f"Cannot locate chunk in source text near position {cursor}"
                )
            start = idx
            end = idx + len(p)
            spans.append((start, end))
            cursor = end
        return spans
