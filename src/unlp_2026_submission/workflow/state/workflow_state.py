from typing import TypedDict

from unlp_2026_submission.entities import SingleAnswerQuestion


class WorkflowState(TypedDict):
    question: SingleAnswerQuestion
    answer: str
    raw_answer: str

    # messages: Annotated[Sequence[BaseMessage], add_messages]
