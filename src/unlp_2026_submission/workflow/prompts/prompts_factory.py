from .en_qa_prompt import EnQAPrompt
from .qa_prompt import QAPrompt
from .chain_of_thought_qa_prompt import ChainOfThoughtQAPrompt
from .qa_prompt_type import QAPromptType


class PromptsFactory:
    _qa_prompt_type: QAPromptType

    def __init__(
            self,
            qa_prompt_type: QAPromptType
    ):
        self._qa_prompt_type = qa_prompt_type

    @staticmethod
    def create(qa_prompt_type: QAPromptType):
        return PromptsFactory(qa_prompt_type=qa_prompt_type)

    def get_qa_prompt(self):
        match self._qa_prompt_type:
            case QAPromptType.SIMPLE:
                return QAPrompt()
            case QAPromptType.SIMPLE_EN:
                return EnQAPrompt()
            case QAPromptType.CHAIN_OF_THOUGHT:
                return ChainOfThoughtQAPrompt()
            case _:
                raise ValueError("Prompt type not found.")
