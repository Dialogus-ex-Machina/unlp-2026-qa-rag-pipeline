from langchain_core.prompts import PromptTemplate

from unlp_2026_submission.entities import Question


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
                "answers",
                "yes_token",
                "no_token"
            ],
            template_format="jinja2"
        )

    def format(
            self,
            question: Question,
            context: str,
            yes_token: str,
            no_token: str,
    ) -> str:
        question_text = question.get('question_text')
        answers = question.get('answers')

        return self._template.format(
            query=question_text,
            context=context,
            answers=answers,
            yes_token=yes_token,
            no_token=no_token
        )
