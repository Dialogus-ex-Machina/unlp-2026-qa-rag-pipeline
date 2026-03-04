import re

from langchain_core.messages import AIMessage

from unlp_2026_submission.rag.qa.nodes.base_node import BaseNode
from unlp_2026_submission.rag.qa.prompts import DomainClassificationPrompt, UkrDomainClassificationPrompt
from unlp_2026_submission.rag.qa.state import QAState
from unlp_2026_submission.models.language_models import LanguageModel
from unlp_2026_submission.entities import QuestionDomain

class LLMDomainRoutingNode(BaseNode):
    language_model: LanguageModel
    prompt: DomainClassificationPrompt

    def __init__(
            self,
            language_model: LanguageModel,
            prompt: DomainClassificationPrompt = UkrDomainClassificationPrompt(),
    ):
        super().__init__()
        self.language_model = language_model
        self.prompt = prompt

    def __call__(self, state: QAState):
        question = state['question']

        prompt = self.prompt.format_messages(question=question)

        response = self.language_model.invoke(prompt)
        predicted_domain, raw_response = self.format_domain_response(response)

        print('Raw predicted domain:', raw_response)
        print('Predicted domain:', predicted_domain)
        print('Correct domain:', question['domain'])

        return {
            'predicted_domain': predicted_domain,
        }

    def format_domain_response(self, response: AIMessage) -> tuple[QuestionDomain, str]:
        raw_response = getattr(response, "content", str(response))
        domain = self._normalize_domain(raw_response)

        return domain, raw_response

    def _normalize_domain(self, text: str) -> QuestionDomain:
        if not text:
            return "other"

        t = text.strip().lower()

        pattern_token = re.compile(r"\b(medicine|sport|other)\b", re.IGNORECASE)
        matches = list(pattern_token.finditer(t))
        if matches:
            return matches[-1].group(1).lower()

        pattern_punct = re.compile(r"(medicine|sport|other)(?=\s*[\)\]\}\.,:;\-—!?\"']|\s*$)", re.IGNORECASE)
        matches = list(pattern_punct.finditer(t))
        if matches:
            return matches[-1].group(1).lower()

        last_token = re.split(r"\s+", t)[-1].strip("()[]{}.,:;—-!?\"'")
        return last_token if last_token in {"medicine", "sport", "other"} else "other"
