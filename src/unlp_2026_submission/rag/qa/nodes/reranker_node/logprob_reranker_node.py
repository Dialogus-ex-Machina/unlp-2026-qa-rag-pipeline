import math

from unlp_2026_submission.entities import RelevantDocument, Question
from unlp_2026_submission.models.language_models import LlamaCppLanguageModel
from unlp_2026_submission.rag.qa.nodes import BaseNode
from unlp_2026_submission.rag.qa.prompts import LogprobRerankerPrompt, UkrLogprobRerankerPrompt
from unlp_2026_submission.rag.qa.state import QAState

class LogprobRerankerNode(BaseNode):
    language_model: LlamaCppLanguageModel
    prompt: LogprobRerankerPrompt
    yes_token: str
    no_token: str

    def __init__(
            self,
            language_model: LlamaCppLanguageModel,
            prompt: LogprobRerankerPrompt = UkrLogprobRerankerPrompt(),
            yes_token = 'Так',
            no_token = 'Ні',
    ):
        self.language_model = language_model
        self.prompt = prompt
        self.yes_token = yes_token
        self.no_token = no_token

    def __call__(self, state: QAState):
        question = state['question']

        relevant_documents = state.get('relevant_documents', [])

        if not relevant_documents:
            print('Relevant context not found. Skipping reranking.')
            return {}

        for document in relevant_documents:
            score = self._score_document(question, document)
            document.relevance_score = score

        reranked_relevant_documents = sorted(
            relevant_documents,
            key=lambda x: x.relevance_score or 0.0, reverse=True
        )

        if not reranked_relevant_documents:
            return {}

        print('Reranked relevant documents:', reranked_relevant_documents)

        return {
            'relevant_documents': reranked_relevant_documents,
        }

    def _score_document(
            self,
            question: Question,
            document: RelevantDocument
    ) -> float:
        prompt = self.prompt.format(
            question=question,
            context=document.text,
            yes_token=self.yes_token,
            no_token=self.no_token,
        )

        result = self.language_model.invoke(
            prompt,
            logprobs=True,
            # TODO update for proper value
            top_logprobs=25,
        )

        top_logprobs = result.response_metadata["logprobs"]["content"][0]["top_logprobs"]

        logp_yes = self._get_logp(top_logprobs, self.yes_token)
        logp_no = self._get_logp(top_logprobs, self.no_token)

        if logp_yes is None or logp_no is None:
            seen = [t["token"] for t in top_logprobs]
            raise ValueError(f"{self.yes_token}/{self.yes_token} not found in top_logprobs. Seen: {seen}")

        score = self._normalized_relevance(logp_yes, logp_no)

        return score

    def _normalized_relevance(self, logp_yes: float, logp_no: float) -> float:
        return 1.0 / (1.0 + math.exp(logp_no - logp_yes))

    def _get_logp(self, top_logprobs, wanted: str) -> float | None:
        for t in top_logprobs:
            if t["token"].strip().lower() in wanted.lower().strip():
                return float(t["logprob"])
        return None


