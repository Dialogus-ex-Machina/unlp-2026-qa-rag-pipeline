from enum import Enum

class QAPromptType(str, Enum):
    SIMPLE = "simple"
    SIMPLE_EN = "simple-en"
    CHAIN_OF_THOUGHT = "cot"
