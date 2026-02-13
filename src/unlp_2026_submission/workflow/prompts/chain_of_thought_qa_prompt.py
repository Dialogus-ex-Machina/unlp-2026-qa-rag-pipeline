from langchain_core.prompts import ChatPromptTemplate

from .base_qa_prompt import BaseQAPrompt

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

class ChainOfThoughtQAPrompt(BaseQAPrompt):
    _template: ChatPromptTemplate

    def __init__(self):
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT
        )
