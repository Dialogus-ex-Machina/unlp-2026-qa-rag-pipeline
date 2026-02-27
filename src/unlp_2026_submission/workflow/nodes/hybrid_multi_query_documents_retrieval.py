import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore

from unlp_2026_submission.entities import RelevantDocument, Question
from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.prompts import MultiQueryPrompt, UkrMultiQueryPrompt
from unlp_2026_submission.workflow.prompts.multi_query import UkrMultiSparseQueryWithAnswersPrompt
from unlp_2026_submission.workflow.state import QAWorkflowState
from unlp_2026_submission.language_models import LanguageModel

# Default RRF constant (k). Higher k reduces impact of top ranks; 60 is standard.
RRF_K = 60


def _doc_key(doc: RelevantDocument) -> tuple[str, str]:
    """Stable key for deduplication across rank lists: (source, page_label)."""
    return (doc.source or "", doc.page_label or "")


class HybridMultiQueryDocumentsRetrievalNode(BaseNode):
    _dense_vector_store: VectorStore
    _sparse_vector_store: VectorStore
    _language_model: LanguageModel
    _dense_prompt: MultiQueryPrompt
    _sparse_prompt: MultiQueryPrompt
    _top_k: int
    _rrf_k: int

    def __init__(
            self,
            dense_vector_store: VectorStore,
            sparse_vector_store: VectorStore,
            language_model: LanguageModel,
            dense_prompt: MultiQueryPrompt = UkrMultiQueryPrompt(),
            sparse_prompt: MultiQueryPrompt = UkrMultiSparseQueryWithAnswersPrompt(),
            top_k: int = 10,
            rrf_k: int = RRF_K,
    ):
        super().__init__()
        self._dense_vector_store = dense_vector_store
        self._sparse_vector_store = sparse_vector_store
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._language_model = language_model
        self._dense_prompt = dense_prompt
        self._sparse_prompt = sparse_prompt

    def __call__(self, state: QAWorkflowState):
        question = state["question"]

        if state.get("relevant_context"):
            print("Relevant context already exists. Skipping context retrieval.")
            return {}

        dense_docs = self._query_dense_rank_lists(question)
        sparse_docs = self._query_sparse_rank_lists(question)

        relevant_documents = self._do_rrf(
            dense_docs=dense_docs,
            sparse_docs=sparse_docs,
        )

        relevant_documents = sorted(
            relevant_documents,
            key=lambda x: x.relevance_score or 0.0,
            reverse=True,
        )[: self._top_k]

        return {"relevant_documents": relevant_documents}

    def _query_dense_rank_lists(self, question: Question) -> list[RelevantDocument]:
        """Generate dense queries via LLM, retrieve per query, return single list with duplicates filtered."""
        queries = self._generate_queries(question, self._dense_prompt)
        print("Dense queries:", queries)

        relevant_documents: list[RelevantDocument] = []
        for query in queries:
            docs_with_score = self._dense_vector_store.similarity_search_with_score(
                query, k=self._top_k
            )
            relevant_documents.extend(RelevantDocument.from_nodes_with_score(docs_with_score))
        return self._filter_unique_documents(relevant_documents)

    def _query_sparse_rank_lists(self, question: Question) -> list[RelevantDocument]:
        """Generate sparse queries via LLM, retrieve per query, return single list with duplicates filtered."""
        queries = self._generate_queries(question, self._sparse_prompt)
        print("Sparse queries:", queries)

        relevant_documents: list[RelevantDocument] = []
        for query in queries:
            docs_with_score = self._sparse_vector_store.similarity_search_with_score(
                query, k=self._top_k
            )
            relevant_documents.extend(RelevantDocument.from_nodes_with_score(docs_with_score))
        return self._filter_unique_documents(relevant_documents)

    def _filter_unique_documents(self, documents: list[RelevantDocument]) -> list[RelevantDocument]:
        """Deduplicate by text; when duplicate, keep higher relevance_score."""
        docs_by_text: dict[str, RelevantDocument] = {}
        for document in documents:
            text = (document.text or "").strip()
            if text not in docs_by_text:
                docs_by_text[text] = document
            else:
                existing = docs_by_text[text]
                if (document.relevance_score or 0.0) > (existing.relevance_score or 0.0):
                    existing.relevance_score = document.relevance_score
        return list(docs_by_text.values())

    def _generate_queries(self, question: Question, prompt: MultiQueryPrompt) -> list[str]:
        """Use LLM to generate multiple queries from the question."""
        chain = (
            self._language_model
            | StrOutputParser()
            | self._clean_queries
        )
        return chain.invoke(prompt.format(question=question))

    def _do_rrf(
            self,
            dense_docs: list[RelevantDocument],
            sparse_docs: list[RelevantDocument],
    ) -> list[RelevantDocument]:
        """
        Reciprocal Rank Fusion (RRF) over the dense and sparse lists.

        Treats dense_docs and sparse_docs as two rank lists (order = rank).
        Score(d) = 1/(k + rank_dense) + 1/(k + rank_sparse), 1-based ranks.
        Documents are matched by (source, page_label); relevance_score is set to RRF score.
        """
        k = self._rrf_k
        rrf_scores: dict[tuple[str, str], float] = {}
        doc_by_key: dict[tuple[str, str], RelevantDocument] = {}

        for one_based_rank, doc in enumerate(dense_docs or [], start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + one_based_rank)
            if key not in doc_by_key:
                doc_by_key[key] = doc

        for one_based_rank, doc in enumerate(sparse_docs or [], start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + one_based_rank)
            if key not in doc_by_key:
                doc_by_key[key] = doc

        result: list[RelevantDocument] = []
        for key, score in rrf_scores.items():
            doc = doc_by_key[key]
            doc.relevance_score = score
            result.append(doc)
        return result

    def _clean_queries(self, output: str) -> list[str]:
        if not output:
            return []

        lines = output.split("\n")

        cleaned = []
        for line in lines:
            q = line.strip()

            # remove numbering like "1. ", "2)", "- ", etc.
            q = re.sub(r"^\d+[\).\s-]*", "", q)

            # skip empty or very short queries
            if not q or len(q) < 3:
                continue

            cleaned.append(q)

        # remove duplicates while preserving order
        seen = set()
        uniq = []
        for q in cleaned:
            if q not in seen:
                seen.add(q)
                uniq.append(q)

        return uniq
