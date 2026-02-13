from langchain_core.prompts import PromptTemplate

RERANKER_PROMPT = (
    "A list of documents is shown below. Each document has a number next to it along "
    "with a summary of the document. A question is also provided. \n"
    "Respond with the numbers of the documents "
    "you should consult to answer the question, in order of relevance, as well \n"
    "as the relevance score. The relevance score is a number from 1-10 based on "
    "how relevant you think the document is to the question.\n"
    "Do not include any documents that are not relevant to the question. \n"
    "Example format: \n"
    "Document 1:\n<summary of document 1>\n\n"
    "Document 2:\n<summary of document 2>\n\n"
    "...\n\n"
    "Document 10:\n<summary of document 10>\n\n"
    "Question: <question>\n"
    "Answer:\n"
    "Doc: 9, Relevance: 7\n"
    "Doc: 3, Relevance: 4\n"
    "Doc: 7, Relevance: 3\n\n"
    "Let's try this now: \n\n"
    "{context}\n"
    "Question: {query}\n"
    "Answer:\n"
)

class RerankerPrompt:
    _template: PromptTemplate

    def __init__(self):
        self._template = PromptTemplate(
            template=RERANKER_PROMPT,
            input_variables=["context", "query"]
        )

    def format(self, query: str, context: str) -> str:
        return self._template.format(query=query, context=context)
