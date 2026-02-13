from unlp_2026_submission.workflow.nodes.base_node import BaseNode
from unlp_2026_submission.workflow.state import QAWorkflowState

class FakeQuestionAnswerNode(BaseNode):
    def __call__(self, state: QAWorkflowState):
        return {
            'raw_answer': 'Fake',
            'answer': 'A',
        }
