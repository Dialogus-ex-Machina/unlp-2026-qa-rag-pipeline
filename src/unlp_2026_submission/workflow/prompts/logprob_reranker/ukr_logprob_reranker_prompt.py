from langchain_core.prompts import PromptTemplate

from .logprob_reranker_prompt import LogprobRerankerPrompt

RERANKER_PROMPT = """
Нижче наведено документ і запитання користувача.
Визнач, чи відповідає документ на це запитання.
У відповідь напиши тільки {yes_token} або {no_token}.
Документ:
{context}
Запитання користувача:
{query}
Відповідь:
"""

class UkrLogprobRerankerPrompt(LogprobRerankerPrompt):
    _template: PromptTemplate

    def __init__(self):
        super().__init__(RERANKER_PROMPT)
