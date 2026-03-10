from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

class SplitsAugmentationPrompt:
    _template: PromptTemplate

    def __init__(
            self,
            prompt_template: str
    ):
        self._template = PromptTemplate(
            template=prompt_template,
            input_variables=["document_splits"],
            template_format="jinja2"
        )

    def format(
            self,
            document_splits: list[Document],
    ) -> str:
        document_splits_content = [doc_split.page_content for doc_split in document_splits]

        return self._template.format(
            document_splits=document_splits_content,
        )
