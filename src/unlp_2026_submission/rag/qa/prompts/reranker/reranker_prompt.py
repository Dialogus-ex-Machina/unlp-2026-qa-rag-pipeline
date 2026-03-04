from langchain_core.prompts import PromptTemplate

class RerankerPrompt:
    _template: PromptTemplate

    def __init__(
            self,
            prompt_template: str
    ):
        self._template = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "query"]
        )

    def format(self, query: str, context: str) -> str:
        return self._template.format(query=query, context=context)
