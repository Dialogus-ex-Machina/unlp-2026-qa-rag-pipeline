from unlp_2026_submission.rag.qa.nodes.base_node import BaseNode
from unlp_2026_submission.rag.qa.state import QAState

class MockQuestionAnswerNode(BaseNode):
    answer: str

    def __init__(self, answer: str = 'A'):
        super().__init__()
        self.answer = answer

    def __call__(self, state: QAState):
        # print('Mock question answer')

        return {
            'raw_answer': self.answer,
            'answer': self.answer,
        }
