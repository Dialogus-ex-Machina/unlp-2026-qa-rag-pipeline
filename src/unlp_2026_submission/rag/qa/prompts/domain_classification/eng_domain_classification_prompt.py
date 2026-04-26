from .domain_classification_prompt import DomainClassificationPrompt

SYSTEM_PROMPT = """
Classify the following question with answers into one of the predefined categories: medicine or sport.
""".strip()

USER_PROMPT = """
Question:
{{ question | trim }}
Answers:
{% set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}
{% for ans in answers -%}
{{ letters[loop.index0] }}. {{ ans }}
{% endfor -%}
""".strip()

class EngDomainClassificationPrompt(DomainClassificationPrompt):
    def __init__(self):
        super().__init__(
            SYSTEM_PROMPT,
            USER_PROMPT
        )
