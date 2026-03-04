import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore

from unlp_2026_submission.entities import RelevantDocument
from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.prompts import MultiQueryPrompt, UkrMultiQueryPrompt
from unlp_2026_submission.workflow.state import QAWorkflowState
from unlp_2026_submission.models.language_models import LanguageModel


class MultiQueryDocumentsRetrievalNode(BaseNode):
    _vector_store: VectorStore
    _language_model: LanguageModel
    _prompt: MultiQueryPrompt
    _top_k: int

    def __init__(
            self,
            vector_store: VectorStore,
            language_model: LanguageModel,
            prompt: MultiQueryPrompt = UkrMultiQueryPrompt(),
            top_k: int = 10,
    ):
        super().__init__()
        self._vector_store = vector_store
        self._top_k = top_k
        self._language_model = language_model
        self._prompt = prompt

    def __call__(self, state: QAWorkflowState):
        question = state['question']

        is_relevant_context_exist = bool(state.get('relevant_context', None))

        if is_relevant_context_exist:
            print('Relevant context already exists. Skipping context retrieval.')
            return {}

        generate_queries = (
                self._language_model
                | StrOutputParser()
                | self._clean_queries
        )

        prompt = self._prompt.format(
            question=question,
        )

        queries = generate_queries.invoke(prompt)
        print('Multiple queries:', queries)

        relevant_documents = []
        for query in queries:
            docs_with_score = self._vector_store.similarity_search_with_score(
                query,
                k=self._top_k
            )

            relevant_documents.extend(RelevantDocument.from_nodes_with_score(docs_with_score))

        relevant_documents = self._filter_unique_documents(relevant_documents)

        relevant_documents = sorted(
            relevant_documents, key=lambda x: x.relevance_score or 0.0, reverse=True
        )[:self._top_k]

        return {
            'relevant_documents': relevant_documents,
        }

    def _filter_unique_documents(self, documents: list[RelevantDocument]):
        docs_by_text: dict[str, RelevantDocument] = {}

        for document in documents:
            text = (document.text or "").strip()

            if text not in docs_by_text:
                docs_by_text[text] = document
            else:
                existing_doc = docs_by_text[text]

                # update score if new one is higher
                if (document.relevance_score or 0.0) > (existing_doc.relevance_score or 0.0):
                    existing_doc.relevance_score = document.relevance_score

        return list(docs_by_text.values())

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
