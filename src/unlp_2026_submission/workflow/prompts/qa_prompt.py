from langchain_core.prompts import ChatPromptTemplate

from .base_qa_prompt import BaseQAPrompt

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

class QAPrompt(BaseQAPrompt):
    _template: ChatPromptTemplate

    def __init__(self):
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
        )
