from langchain_core.prompts import ChatPromptTemplate

from .base_qa_prompt import BaseQAPrompt

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

class EngQAPrompt(BaseQAPrompt):
    _template: ChatPromptTemplate

    def __init__(self):
        super().__init__(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT,
        )
