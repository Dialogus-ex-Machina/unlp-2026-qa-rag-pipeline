from langchain_core.prompts import PromptTemplate

from .reranker_prompt import RerankerPrompt


RERANKER_PROMPT = """
A list of documents is shown below. Each document has a number next to it along with a content of the document. A question is also provided.
Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well as the relevance score.
The relevance score is a number from 1-10 based on how relevant you think the document is to the question.
Do not include any documents that are not relevant to the question.
Example format:
Document 1:
<summary of document 1>

Document 2:
<summary of document 2>

...

Document 10:
<summary of document 10>

Question: <question>
Answer:
Doc: 9, Relevance: 7
Doc: 3, Relevance: 4
Doc: 7, Relevance: 3

Let's try this now:

{context}
Question: {query}
Answer:
"""

class EngRerankerPrompt(RerankerPrompt):
    _template: PromptTemplate

    def __init__(self):
        super().__init__(RERANKER_PROMPT)
