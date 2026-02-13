from abc import ABC, abstractmethod

from unlp_2026_submission.workflow.state import QAWorkflowState

class BaseNode(ABC):
    @abstractmethod
    def __call__(self, state: QAWorkflowState):
        pass
