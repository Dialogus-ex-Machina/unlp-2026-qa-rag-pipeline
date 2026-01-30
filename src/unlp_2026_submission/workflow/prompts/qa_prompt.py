from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from unlp_2026_submission.entities import Question, DocumentPage

SYSTEM_PROMPT = """
Ти — корисний асистент для вирішення завдань з вибором однієї правильної відповіді.
Відповідай лише літерою варіанту (A, B, C, D, E або F).
{% if context %}
Відповідай **тільки** на основі наведеного контексту.
Контекст:
{{ context }}
{% endif %}
""".strip()
# TODO remove ABCDEFGHIJKLMNOPQRSTUVWXYZ from prompt
USER_PROMPT = """
Завдання:
{{ question | trim }}
Варіанти:
{% set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}
{% for ans in answers -%}
{{ letters[loop.index0] }}. {{ ans }}
{% endfor -%}
Відповідай лише літерою варіанту.
""".strip()

class QAPrompt:
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
