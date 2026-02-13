from typing import TypedDict

class Question(TypedDict):
    question_id: int | str
    question_text: str
    answers: list[str]

    domain: str | None
    n_pages: int | None
    correct_answer: str | None
    doc_id: str | None
    page_num: int | None

class QuestionWithContext(Question):
    domain: str
    n_pages: int
    correct_answer: str
    doc_text: str
    doc_id: str
    page_num: int
