from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from unlp_2026_submission.entities import Question


class DomainClassificationPrompt:
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
    ) -> list[BaseMessage]:
        question_text = question.get('question_text')
        answers = question.get('answers')

        return self._template.format_messages(
            question=question_text,
            answers=answers,
        )
