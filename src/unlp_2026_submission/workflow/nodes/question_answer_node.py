from unlp_2026_submission.config import Config
from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.prompts import QuestionAnswerPrompt
from unlp_2026_submission.workflow.state import WorkflowState
from unlp_2026_submission.language_models import LanguageModel

QUESTION_ANSWER_NODE_NAME = 'question_answer_node'

class QuestionAnswerNode(BaseNode):
    def __init__(
            self,
            config: Config,
            language_model: LanguageModel
    ):
        super().__init__(
            name=QUESTION_ANSWER_NODE_NAME,
            config=config,
            language_model=language_model
        )

    def __call__(self, state: WorkflowState):
        question = state['question']

        prompt = QuestionAnswerPrompt().format_messages(
            question=question
        )
        response = self.language_model.invoke(prompt)

        formatted_response = self.format_single_answer_response(response)

        print('raw_answer:', formatted_response['raw_answer'])
        print('answer:', formatted_response['answer'])
        print('correct_answer:', question['correct_answer'])

        return {
            **formatted_response,
        }
