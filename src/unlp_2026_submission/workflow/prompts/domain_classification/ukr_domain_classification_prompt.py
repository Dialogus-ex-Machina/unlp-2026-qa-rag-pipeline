from .domain_classification_prompt import DomainClassificationPrompt

SYSTEM_PROMPT = """
Класифікуй наведене запитання з варіантами відповідей в одну з категорій: medicine, sport або other.
""".strip()

USER_PROMPT = """
Запитання:
{{ question | trim }}
Варіанти відповідей:
{% set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}
{% for ans in answers -%}
{{ letters[loop.index0] }}. {{ ans }}
{% endfor -%}
""".strip()

class UkrDomainClassificationPrompt(DomainClassificationPrompt):
    def __init__(self):
        super().__init__(
            SYSTEM_PROMPT,
            USER_PROMPT
        )
