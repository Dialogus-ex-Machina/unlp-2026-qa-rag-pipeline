from langchain_core.prompts import PromptTemplate

from unlp_2026_submission.entities import Question


class MultiQueryPrompt:
    _template: PromptTemplate

    def __init__(
            self,
            prompt_template: str
    ):
        self._template = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "answers"],
            template_format="jinja2",
        )

    def format(
            self,
            question: Question,
    ) -> str:
        question_text = question['question_text']
        answers = question['answers']

        return self._template.format(
            question=question_text,
            answers=answers
        )

    @property
    def template(self):
        return self._template
