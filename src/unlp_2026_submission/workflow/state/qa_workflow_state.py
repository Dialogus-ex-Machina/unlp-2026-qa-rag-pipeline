from typing import TypedDict
from unlp_2026_submission.entities import Question, RelevantDocument, QuestionDomain


class QAWorkflowState(TypedDict):
    question: Question
    relevant_documents: list[RelevantDocument]

    predicted_domain: QuestionDomain
    relevant_context: str
    relevant_document_id: str
    relevant_document_page_num: int

    answer: str
    raw_answer: str
