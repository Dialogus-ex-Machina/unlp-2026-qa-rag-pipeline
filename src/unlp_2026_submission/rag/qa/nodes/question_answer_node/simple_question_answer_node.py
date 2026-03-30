import re

from langchain_core.messages import AIMessage

from unlp_2026_submission.rag.qa.nodes.base_node import BaseNode
from unlp_2026_submission.rag.qa.prompts.qa import BaseQAPrompt
from unlp_2026_submission.rag.qa.state import QAState
from unlp_2026_submission.models.language_models import LanguageModel


class SimpleQuestionAnswerNode(BaseNode):
    def __init__(
            self,
            language_model: LanguageModel,
            prompt: BaseQAPrompt,
    ):
        super().__init__()
        self.language_model = language_model
        self.prompt = prompt

    def __call__(self, state: QAState):
        question = state['question']
        current_context = state['relevant_context']

        while True:
            prompt = self.prompt.format_messages(
                question=question,
                relevant_context=current_context
            )

            try:
                response = self.language_model.invoke(prompt)

                formatted_response = self.format_single_answer_response(response)

                return {
                    **formatted_response,
                }
            except Exception:
                # If we've reduced the context to nothing and it still fails,
                # return the fallback to prevent an infinite loop.
                if not current_context:
                    return {
                        'raw_answer': 'A',
                        'answer': 'A',
                    }

                # Keep top 85%, chopping off the bottom 15%.
                # min() ensures we always drop at least 1 char/item to prevent infinite loops.
                target_length = min(len(current_context) - 1, int(len(current_context) * 0.85))
                # Ensure target_length doesn't drop below 0
                target_length = max(0, target_length)

                current_context = current_context[:target_length]

    def format_single_answer_response(self, response: AIMessage):
        raw_answer = getattr(response, "content", str(response))
        answer = self._normalize_choice(raw_answer)

        return {
            'raw_answer': raw_answer,
            'answer': answer,
        }

    def _normalize_choice(self, text: str) -> str:
        """
        Повертає 'A'/'B'/'C'/'D' якщо знаходить.
        Працює з: 'A', 'A)', 'Answer: C', ' C \n', 'Option D', тощо.
        """
        if not text:
            return ""

        t = text.upper()

        # Шукаємо з КІНЦЯ
        rev = t[::-1]
        m = re.search(r"\b([A-F])\b", rev)
        if m:
            return m.group(1)

        # fallback: якщо модель написала щось типу "A)" або "A."
        m = re.search(r"([A-F])(?=\s*[\)\.\:\-])", rev)
        # TODO add fallback for ukrainian letters
        return m.group(1) if m else rev.strip()[:1]