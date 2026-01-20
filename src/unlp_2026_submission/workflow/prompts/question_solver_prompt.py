from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from unlp_2026_submission.entities import SingleAnswerQuestion

SYSTEM_PROMPT = """
Ти — корисний асистент для вирішення завдань з вибором однієї правильної відповіді.
Відповідай лише літерою варіанту (A, B, C, D, E або F).
""".strip()
# TODO remove ABCDEFGHIJKLMNOPQRSTUVWXYZ from prompt
USER_PROMPT = """
Завдання:
{{ question | trim }}
Варіанти:
{% set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}
{% for ans in answers -%}
{{ letters[loop.index0] }}. {{ ans }}
{% endfor -%}
Відповідай лише літерою варіанту.
""".strip()

SYSTEM_PROMPT_EN = """
You are a helpful assistant for multiple-choice questions.
Reply with only the option letter (A, B, C, D, E or F).
""".strip()

USER_PROMPT_EN = """
Question:
{{ question | trim }}
Options:
{% set letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" -%}
{% for ans in answers -%}
{{ letters[loop.index0] }}. {{ ans }}
{% endfor -%}
Respond with only the option letter.
""".strip()

class QuestionSolverPrompt:
    _template: ChatPromptTemplate

    def __init__(self):
        self._template = ChatPromptTemplate.from_messages(
            messages=[
                ("system", SYSTEM_PROMPT_EN),
                ("human", USER_PROMPT_EN),
            ],
            template_format="jinja2"
        )

    def format_messages(self, question: SingleAnswerQuestion) -> list[BaseMessage]:
        question_text = question.get('question_text')
        answers = question.get('answers')

        return self._template.format_messages(
            question=question_text,
            answers=answers
        )
