from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from unlp_2026_submission.entities import Question, DocumentPage

SYSTEM_PROMPT = """
Ти — асистент для розв'язання завдань з вибором однієї правильної відповіді.

Твої дії:
- покроково знайди відповідь відповідно до контексту
- перевір свої міркування
- звір результат з варіантами відповідей
- обери правильний варіант

Важливо:
ти МАЄШ виводити свої міркування.
В кінці ти ПОВИНЕН вивести літеру варіанту: (A, B, C, D, E або F).

{% if context %}
Відповідай **тільки** на основі наведеного контексту.
Контекст:
{{ context }}
{% endif %}
""".strip()

USER_PROMPT = """
Завдання:
{{ question | trim }}
Варіанти:
{% for ans in answers -%}
{{ loop.index }}. {{ ans }}
{% endfor -%}
Уважно подумай і перевір свій результат.
Додай в кінці літеру правильного варіанту.

Важливо:
Якщо жоден варіант точно не відповідає результату, перевір усе ще раз.
Один із варіантів завжди є правильним.
""".strip()

class ChainOfThoughtQAPrompt:
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
