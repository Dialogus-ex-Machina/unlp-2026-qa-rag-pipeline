from .domain_classification import (
    EngDomainClassificationPrompt,
    UkrDomainClassificationPrompt,
    DomainClassificationPromptType
)
from .qa import (
    QAPromptType,
    UkrQAPrompt,
    EngQAPrompt,
    UkrChainOfThoughtQAPrompt,
)

class PromptsFactory:
    @staticmethod
    def get_qa_prompt(
            prompt_type: QAPromptType
    ):
        match prompt_type:
            case QAPromptType.UKR:
                return UkrQAPrompt()
            case QAPromptType.ENG:
                return EngQAPrompt()
            case QAPromptType.UKR_CHAIN_OF_THOUGHT:
                return UkrChainOfThoughtQAPrompt()
            case _:
                raise ValueError("Prompt type not found.")

    @staticmethod
    def get_domain_classification_prompt(
            prompt_type: DomainClassificationPromptType
    ):
        match prompt_type:
            case DomainClassificationPromptType.ENG:
                return EngDomainClassificationPrompt()
            case DomainClassificationPromptType.UKR:
                return UkrDomainClassificationPrompt()
            case _:
                raise ValueError("Prompt type not found.")
