from langchain_core.prompts import PromptTemplate

from .reranker_prompt import RerankerPrompt

RERANKER_PROMPT = """
Нижче наведено список документів. Біля кожного документа вказано його номер та зміст. Також надано запитання.
У відповіді зазнач номери документів, які слід використати для відповіді на запитання, у порядку релевантності, а також вкажи оцінку релевантності.
Оцінка релевантності — це число від 1 до 10, яке відображає, наскільки, на твою думку, документ є релевантним до запитання.
Не включай у відповідь документи, які не є релевантними до запитання.
Приклад формату:
Document 1:
<зміст документа 1>

Document 2:
<зміст документа 2>

...

Document 10:
<зміст документа 10>

Запитання: <запитання>
Відповідь:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3

Тепер твоя черга:

{context}
Запитання: {query}
Відповідь:
"""

class UkrRerankerPrompt(RerankerPrompt):
    _template: PromptTemplate

    def __init__(self):
        super().__init__(RERANKER_PROMPT)
