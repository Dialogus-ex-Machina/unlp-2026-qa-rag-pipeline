from abc import ABC, abstractmethod
import re

from langchain_core.messages import AIMessage

from unlp_2026_submission.language_models import LanguageModel
from unlp_2026_submission.config import Config
from unlp_2026_submission.workflow.state import WorkflowState

class BaseNode(ABC):
    name: str
    config: Config
    language_model: LanguageModel

    def __init__(
            self,
            name: str,
            config: Config,
            language_model: LanguageModel,
    ):
        self.name = name
        self.config = config
        self.language_model = language_model

    @abstractmethod
    def __call__(self, state: WorkflowState):
        pass

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
