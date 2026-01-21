from typing import TypedDict

class SingleAnswerQuestion(TypedDict):
    question_id: int
    question_text: str
    answers: list[str]

    domain: str | None
    n_pages: int | None
    correct_answer: str | None
    doc_id: str | None
    page_num: int | None
