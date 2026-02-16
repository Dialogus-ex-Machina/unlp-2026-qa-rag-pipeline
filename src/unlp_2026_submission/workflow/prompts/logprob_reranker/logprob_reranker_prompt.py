from langchain_core.prompts import PromptTemplate

class LogprobRerankerPrompt:
    _template: PromptTemplate

    def __init__(
            self,
            prompt_template: str
    ):
        self._template = PromptTemplate(
            template=prompt_template,
            input_variables=[
                "context",
                "query",
                "yes_token",
                "no_token"
            ]
        )

    def format(
            self, query: str,
            context: str,
            yes_token: str,
            no_token: str,
    ) -> str:
        return self._template.format(
            query=query,
            context=context,
            yes_token=yes_token,
            no_token=no_token
        )
