from enum import Enum

class EvaluationDatasetName(str, Enum):
    FULL_QA = "full-qa"
    SPORT_QA = "sport-qa"
    MEDICINE_QA = "medicine-qa"
    FULL_QA_WITH_CONTEXT = "full-qa-with-context"
    SPORT_QA_WITH_CONTEXT = "sport-qa-with-context"
    MEDICINE_QA_WITH_CONTEXT = "medicine-qa-with-context"
