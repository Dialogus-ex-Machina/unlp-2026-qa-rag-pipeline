from typing import List, Sequence, Any, Tuple
import numpy as np
import re
import copy
import tiktoken

from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core.node_parser import SentenceSplitter
from unlp_2026_submission.embeddings import EmbeddingsModel

# - "1. " (арабські)
# - "IV. " (латинські римські)
# - "І. " (кирилична "І", часто в укр. документах)
LIST_MARKER_RE = re.compile(r"(?m)^(\s*)(\d+|[IVXLCDM]+|[І]+)\.\s+")

DOT_PLACEHOLDER = "∯"

def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens

class ClusterSemanticSplitter(BaseDocumentTransformer):
    def __init__(
            self,
            embeddings_model: EmbeddingsModel,
            max_chunk_size: int,
            min_chunk_size: int,
            batch_size=10,
    ):
        self.splitter = SentenceSplitter(
            chunk_size=min_chunk_size,
            chunk_overlap=0,
        )

        self._chunk_size = max_chunk_size
        self.max_cluster = max_chunk_size // min_chunk_size
        self.embeddings_model = embeddings_model
        self.batch_size = batch_size


    def transform_documents(
            self,
            documents: Sequence[Document],
            **kwargs: Any
    ) -> Sequence[Document]:
        transformed_documents = []

        for document in documents:
            split_documents = self.split_document(document)
            transformed_documents.extend(split_documents)

        return transformed_documents

    def split_document(self, document: Document) -> list[Document]:
        split_documents = []

        chunks = self._split_text(document.page_content)

        for chunk_text, start, end in chunks:
            doc_metadata = copy.deepcopy(document.metadata)
            doc_metadata["start_index"] = start
            doc_metadata["end_index"] = end
            split_documents.append(Document(page_content=chunk_text, metadata=doc_metadata))

        return split_documents

    def _split_text(self, text: str) -> List[Tuple[str, int, int]]:
        preprocessed_text = self._preprocess_text(text)

        sentences = self.splitter.split_text(preprocessed_text)
        sent_spans = self._find_ordered_spans(preprocessed_text, sentences)

        sentences_unmasked = self._postprocess_chunks(sentences)

        similarity_matrix = self._get_similarity_matrix(sentences_unmasked)
        clusters = self._optimal_segmentation(similarity_matrix, max_cluster_size=self.max_cluster)

        out: List[Tuple[str, int, int]] = []
        for s_idx, e_idx in clusters:
            chunk_start = sent_spans[s_idx][0]
            chunk_end = sent_spans[e_idx][1]

            # safest: take exact slice from ORIGINAL text (not from joined sentences)
            chunk_text = text[chunk_start:chunk_end]

            out.append((chunk_text, chunk_start, chunk_end))

        return out

    def _find_ordered_spans(
            self,
            full_text: str,
            parts: List[str]
    ) -> List[Tuple[int, int]]:
        """
        Finds each part in full_text in order, starting each search after the previous match.
        Returns [(start, end_exclusive), ...]
        """
        spans: List[Tuple[int, int]] = []
        cursor = 0

        for p in parts:
            if not p:
                spans.append((cursor, cursor))
                continue

            idx = full_text.find(p, cursor)
            if idx == -1:
                # If this happens, splitter output doesn't match the source text exactly.
                # Common fix: normalize whitespace in both strings, or use SequenceMatcher fallback.
                raise ValueError(f"Cannot locate sentence chunk in source text near position {cursor}")

            start = idx
            end = idx + len(p)
            spans.append((start, end))
            cursor = end

        return spans

    def _get_similarity_matrix(self, sentences: list[str]):
        n_sentences = len(sentences)
        embedding_matrix = None

        for i in range(0, n_sentences, self.batch_size):
            batch_sentences = sentences[i:i + self.batch_size]
            embeddings = self.embeddings_model.embed_documents(batch_sentences)
            embeddings = [np.array(embedding, dtype=np.float32) for embedding in embeddings]

            # Convert embeddings list of lists to numpy array
            batch_embedding_matrix = np.array(embeddings)

            # Append the batch embedding matrix to the main embedding matrix
            if embedding_matrix is None:
                embedding_matrix = batch_embedding_matrix
            else:
                embedding_matrix = np.concatenate((embedding_matrix, batch_embedding_matrix), axis=0)

        similarity_matrix = np.dot(embedding_matrix, embedding_matrix.T)

        return similarity_matrix

    def _calculate_reward(self, matrix, start, end):
        sub_matrix = matrix[start:end + 1, start:end + 1]
        return np.sum(sub_matrix)

    def _optimal_segmentation(self, matrix, max_cluster_size):
        mean_value = np.mean(matrix[np.triu_indices(matrix.shape[0], k=1)])
        matrix = matrix - mean_value  # Normalize the matrix
        np.fill_diagonal(matrix, 0)  # Set diagonal to 1 to avoid trivial solutions

        n = matrix.shape[0]
        dp = np.zeros(n)
        segmentation = np.zeros(n, dtype=int)

        for i in range(n):
            for size in range(1, max_cluster_size + 1):
                if i - size + 1 >= 0:
                    reward = self._calculate_reward(matrix, i - size + 1, i)
                    # Adjust reward based on local density
                    adjusted_reward = reward
                    if i - size >= 0:
                        adjusted_reward += dp[i - size]
                    if adjusted_reward > dp[i]:
                        dp[i] = adjusted_reward
                        segmentation[i] = i - size + 1

        clusters = []
        i = n - 1
        while i >= 0:
            start = segmentation[i]
            clusters.append((start, i))
            i = start - 1

        clusters.reverse()
        return clusters

    def _preprocess_text(self, text: str) -> str:
        return self._mask_list_markers(text)

    def _postprocess_chunks(self, sentences: list[str]) -> list[str]:
        return [self._unmask_list_markers(s) for s in sentences]

    def _mask_list_markers(self, text: str) -> str:
        return LIST_MARKER_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{DOT_PLACEHOLDER} ", text)

    def _unmask_list_markers(self, text: str) -> str:
        return re.sub(
            rf"(\b\d+|\b[IVXLCDM]+|\b[І]+){re.escape(DOT_PLACEHOLDER)}\s",
            r"\1. ",
            text,
        )