from abc import ABC, abstractmethod

from unlp_2026_submission.language_models import LanguageModel
from unlp_2026_submission.config import Config
from unlp_2026_submission.workflow.state import WorkflowState


class BaseNode(ABC):
    name: str
    config: Config
    language_model: LanguageModel

    def __init__(
            self,
            name: str,
            config: Config,
            language_model: LanguageModel,
    ):
        self.name = name
        self.config = config
        self.language_model = language_model

    @abstractmethod
    async def __call__(self, state: WorkflowState):
        pass
