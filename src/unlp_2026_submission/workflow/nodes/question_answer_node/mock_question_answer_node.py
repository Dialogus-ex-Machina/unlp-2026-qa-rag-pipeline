from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.state import QAWorkflowState

class MockQuestionAnswerNode(BaseNode):
    answer: str

    def __init__(self, answer: str = 'A'):
        super().__init__()
        self.answer = answer

    def __call__(self, state: QAWorkflowState):
        print('Mock question answer')

        return {
            'raw_answer': self.answer,
            'answer': self.answer,
        }
