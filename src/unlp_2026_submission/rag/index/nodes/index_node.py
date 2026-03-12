from typing import Protocol

from unlp_2026_submission.rag.index import IndexState


class IndexNode(Protocol):
    def __call__(self, state: IndexState) -> IndexState: ...
