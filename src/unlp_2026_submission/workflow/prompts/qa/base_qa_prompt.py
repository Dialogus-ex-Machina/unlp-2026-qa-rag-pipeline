from abc import ABC

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from unlp_2026_submission.entities import Question


class BaseQAPrompt(ABC):
    _template: ChatPromptTemplate

    def __init__(
            self,
            system_prompt: str,
            user_prompt: str
    ):
        self._template = ChatPromptTemplate.from_messages(
            messages=[
                ("system", system_prompt),
                ("human", user_prompt),
            ],
            template_format="jinja2"
        )

    def format_messages(
            self,
            question: Question,
            relevant_context: str | None = None
    ) -> list[BaseMessage]:
        question_text = question.get('question_text')
        answers = question.get('answers')

        context = relevant_context if relevant_context else ''

        return self._template.format_messages(
            question=question_text,
            answers=answers,
            context=context
        )
