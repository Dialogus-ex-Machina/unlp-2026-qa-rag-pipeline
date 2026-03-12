from langchain_core.prompts import PromptTemplate

class SplitsAugmentationPrompt:
    _template: PromptTemplate

    def __init__(
            self,
            prompt_template: str
    ):
        self._template = PromptTemplate(
            template=prompt_template,
            input_variables=["chunk_content", "doc_content"]
        )

    def format(
            self,
            chunk_content: str,
            doc_content: str
    ) -> str:
        return self._template.format(
            chunk_content=chunk_content,
            doc_content=doc_content
        )
