from typing import List
from abc import ABC, abstractmethod

from unlp_2026_submission.entities import RelevantDocument


class RerankerModel(ABC):
    @abstractmethod
    def rerank(
            self,
            query: str,
            documents: List[RelevantDocument],
    ) -> List[RelevantDocument]:
        pass