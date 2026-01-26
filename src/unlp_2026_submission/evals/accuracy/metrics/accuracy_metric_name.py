from enum import Enum

class AccuracyMetricName(str, Enum):
    ANSWERS = "answers"
    COMPOSITE = "composite"
    DOCUMENT_SOURCES = "doc-sources"
    DOCUMENT_SOURCE_PAGES = "doc-source-pages"
