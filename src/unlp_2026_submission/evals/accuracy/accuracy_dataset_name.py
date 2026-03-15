from enum import Enum

class AccuracyDatasetName(str, Enum):
    FULL = "full"
    SPORT = "sport"
    MEDICINE = "medicine"
    OTHER = "other"
