from typing import TypedDict, NotRequired

class Question(TypedDict):
    question_id: int | str
    question_text: str
    answers: list[str]

    domain: NotRequired[str]
    n_pages: NotRequired[int]
    correct_answer: NotRequired[str]
    doc_text: NotRequired[str]
    doc_id: NotRequired[str]
    page_num: NotRequired[int]

class QuestionWithContext(TypedDict):
    question_id: int | str
    question_text: str
    answers: list[str]

    domain: NotRequired[str]
    n_pages: NotRequired[int]
    correct_answer: NotRequired[str]
    doc_text: str
    doc_id: str
    page_num: int
