from typing import TypedDict

class SingleAnswerQuestion(TypedDict):
    question_id: str
    question_text: str
    answers: list[str]
    correct_answer: str

    domain: str | None
    n_pages: int | None
    correct_answer: int | None
    doc_id: str | None
    page_num: int | None
