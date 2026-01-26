from enum import Enum

class AccuracyMetricName(str, Enum):
    ANSWERS = "answers-accuracy"
    COMPOSITE = "composite-accuracy"
    DOCUMENT_SOURCES = "doc-sources-accuracy"
    DOCUMENT_SOURCE_PAGES = "doc-source-pages-accuracy"
