from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from unlp_2026_submission.entities import Question, DocumentPage

SYSTEM_PROMPT = """
You are a helpful assistant for multiple-choice questions.
Reply with only the option letter (A, B, C, D, E or F).
{% if context %}
Answer **only** based on the context below.  
Context:
{{ context }}
{% endif %}
""".strip()

USER_PROMPT = """
Question:
{{ question | trim }}
Options:
{% set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}
{% for ans in answers -%}
{{ letters[loop.index0] }}. {{ ans }}
{% endfor -%}
Respond with only the option letter.
""".strip()

class EnQAPrompt:
    _template: ChatPromptTemplate

    def __init__(self):
        self._template = ChatPromptTemplate.from_messages(
            messages=[
                ("system", SYSTEM_PROMPT),
                ("human", USER_PROMPT),
            ],
            template_format="jinja2"
        )

    def format_messages(
            self,
            question: Question,
            document_page: DocumentPage | None = None
    ) -> list[BaseMessage]:
        question_text = question.get('question_text')
        answers = question.get('answers')

        context = document_page.text if document_page else ''

        return self._template.format_messages(
            question=question_text,
            answers=answers,
            context=context
        )
