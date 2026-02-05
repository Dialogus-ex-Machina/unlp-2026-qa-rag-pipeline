from typing import Protocol

from merlin.rag.index import IndexState


class IndexNode(Protocol):
    def __call__(self, state: IndexState) -> IndexState: ...
