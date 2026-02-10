from . import UADomainClassificationPrompt
from .en_domain_classification_prompt import ENDomainClassificationPrompt
from .domain_classification_prompt_type import DomainClassificationPromptType
from .en_qa_prompt import EnQAPrompt
from .qa_prompt import QAPrompt
from .chain_of_thought_qa_prompt import ChainOfThoughtQAPrompt
from .qa_prompt_type import QAPromptType


class PromptsFactory:
    @staticmethod
    def get_qa_prompt(
            prompt_type: QAPromptType
    ):
        match prompt_type:
            case QAPromptType.SIMPLE:
                return QAPrompt()
            case QAPromptType.SIMPLE_EN:
                return EnQAPrompt()
            case QAPromptType.CHAIN_OF_THOUGHT:
                return ChainOfThoughtQAPrompt()
            case _:
                raise ValueError("Prompt type not found.")

    @staticmethod
    def get_domain_classification_prompt(
            prompt_type: DomainClassificationPromptType
    ):
        match prompt_type:
            case DomainClassificationPromptType.SIMPLE_EN:
                return ENDomainClassificationPrompt()
            case DomainClassificationPromptType.SIMPLE_UA:
                return UADomainClassificationPrompt()
            case _:
                raise ValueError("Prompt type not found.")
