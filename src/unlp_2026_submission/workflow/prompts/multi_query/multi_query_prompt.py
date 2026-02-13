from langchain_core.prompts import PromptTemplate

class MultiQueryPrompt:
    _template: PromptTemplate

    def __init__(
            self,
            prompt_template: str
    ):
        self._template = PromptTemplate(
            template=prompt_template,
            input_variables=["query"]
        )

    def format(self, query: str) -> str:
        return self._template.format(query=query)

    @property
    def template(self):
        return self._template
