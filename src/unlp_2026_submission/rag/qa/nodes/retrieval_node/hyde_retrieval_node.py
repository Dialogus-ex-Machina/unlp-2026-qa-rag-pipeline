from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from unlp_2026_submission.entities import RelevantDocument
from unlp_2026_submission.models.language_models import LanguageModel
from unlp_2026_submission.rag.qa.nodes import BaseNode
from unlp_2026_submission.rag.qa.prompts import HydePrompt, UkrHydePrompt
from unlp_2026_submission.rag.qa.state import QAWorkflowState

class HydeRetrievalNode(BaseNode):
    _language_model: LanguageModel
    _vector_store: VectorStore
    _top_k: int = 10
    _prompt: HydePrompt

    def __init__(
            self,
            language_model: LanguageModel,
            vector_store: VectorStore,
            top_k: int = 10,
            prompt: HydePrompt = UkrHydePrompt(),
    ):
        self.language_model = language_model
        self._vector_store = vector_store
        self._top_k = top_k
        self._prompt = prompt

    def __call__(self, state: QAWorkflowState):
        question = state['question']

        is_relevant_context_exist = bool(state.get('relevant_context', None))

        if is_relevant_context_exist:
            print('Relevant context already exists. Skipping context retrieval.')
            return {}

        chain = self._prompt.template | self.language_model | StrOutputParser()

        hyde_query = chain.invoke({ 'query': question['question_text'] })

        print('Raw query', question['question_text'])
        print('Rephrase query', hyde_query)

        docs_with_score = self._vector_store.similarity_search_with_score(
            hyde_query,
            k=self._top_k
        )

        relevant_documents = RelevantDocument.from_nodes_with_score(docs_with_score)

        return {
            'relevant_documents': relevant_documents,
        }

