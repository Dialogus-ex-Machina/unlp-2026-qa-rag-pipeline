from langchain_core.prompts import PromptTemplate

from .logprob_reranker_prompt import LogprobRerankerPrompt

RERANKER_PROMPT = """
Нижче наведено документ і запитання користувача з варіантами відповідей.
Визнач, чи відповідає документ на це запитання.
У відповідь напиши лише {{ yes_token | trim }} або {{ no_token | trim }}.
Документ:
{{ context | trim }}
Запитання користувача:
{{ query | trim }}
Варіанти:
{% set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}
{% for ans in answers -%}
{{ letters[loop.index0] }}. {{ ans }}
{% endfor -%}

Відповідь:
""".strip()

class UkrLogprobRerankerPrompt(LogprobRerankerPrompt):
    _template: PromptTemplate

    def __init__(self):
        super().__init__(RERANKER_PROMPT)
