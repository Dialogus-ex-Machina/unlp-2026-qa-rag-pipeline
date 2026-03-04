from unlp_2026_submission.rag.qa.nodes.base_node import BaseNode
from unlp_2026_submission.rag.qa.state import QAWorkflowState
from unlp_2026_submission.entities import QuestionDomain

class MockDomainRoutingNode(BaseNode):
    domain: QuestionDomain

    def __init__(
            self,
            domain: QuestionDomain = 'other',
    ):
        super().__init__()
        self.domain = domain


    def __call__(self, state: QAWorkflowState):
        return {
            'predicted_domain': self.domain,
        }
