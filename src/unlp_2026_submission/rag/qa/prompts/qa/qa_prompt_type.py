from enum import Enum

class QAPromptType(str, Enum):
    UKR = "ukr"
    ENG = "eng"
    UKR_CHAIN_OF_THOUGHT = "ukr-cot"
