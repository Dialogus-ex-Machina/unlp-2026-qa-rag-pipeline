from typing import TypedDict

from unlp_2026_submission.entities import SingleAnswerQuestion, DocumentPage


class WorkflowState(TypedDict):
    question: SingleAnswerQuestion

    answer: str
    raw_answer: str
    reference_document_page: DocumentPage | None
    reference_document_id: str
    reference_document_page_num: int

    # messages: Annotated[Sequence[BaseMessage], add_messages]
