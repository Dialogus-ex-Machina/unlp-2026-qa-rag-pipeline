from abc import ABC, abstractmethod

from unlp_2026_submission.workflow.state import QAWorkflowState

class BaseNode(ABC):
    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, state: QAWorkflowState):
        pass
