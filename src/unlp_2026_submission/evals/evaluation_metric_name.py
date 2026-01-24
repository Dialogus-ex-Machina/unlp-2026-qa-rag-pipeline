from enum import Enum

class EvaluationMetricName(str, Enum):
    ANSWER_ACCURACY = "answer-accuracy"
    ANSWER_FAITHFULNESS = "answer-faithfulness"
    COMPOSITE_ACCURACY = "composite-accuracy"
    DOCUMENT_SOURCE_ACCURACY = "doc-source-accuracy"
    DOCUMENT_SOURCE_PAGE_ACCURACY = "doc-source-page-accuracy"
