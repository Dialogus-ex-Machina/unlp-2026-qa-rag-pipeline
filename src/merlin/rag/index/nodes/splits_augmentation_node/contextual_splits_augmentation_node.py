from collections import defaultdict
from typing import Optional, Callable

from langchain_core.documents import Document

from merlin.rag.index.index_state import IndexState
from unlp_2026_submission.models.language_models import LanguageModel
from unlp_2026_submission.workflow.prompts import UkrContextualSplitsAugmentationPrompt, SplitsAugmentationPrompt


class ContextualSplitsAugmentationNode:
    language_model: LanguageModel
    sliding_window: int
    on_success: Optional[Callable[[], None]] = None,
    prompt: SplitsAugmentationPrompt

    def __init__(
            self,
            language_model: LanguageModel,
            sliding_window: int = 3,
            on_success: Optional[Callable[[], None]] = None,
            prompt: SplitsAugmentationPrompt = UkrContextualSplitsAugmentationPrompt()
    ):
        self.language_model = language_model
        self.sliding_window = max(1, int(sliding_window))
        self.on_success = on_success
        self.prompt = prompt

    def __call__(self, state) -> IndexState:
        splits: list[Document] = state["splits"]

        # 1) group indices by doc key (preserve original order)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, d in enumerate(splits):
            groups[self.get_doc_key(d)].append(idx)

        for _, idxs in groups.items():
            group_docs = [splits[j] for j in idxs]

            for local_i, global_idx in enumerate(idxs):
                split = splits[global_idx]
                window_docs = self._get_window(group_docs, local_i)
                context = self._get_context_for_split(window_docs, split)

                print('Added context:', context)

                split.page_content = f"{split.page_content}\n\n{context}"

        if self.on_success:
            self.on_success()

        return { "splits": splits }


    def get_doc_key(self, d: Document) -> str:
        return str(d.metadata.get("source") or d.metadata.get("file_name") or d.metadata.get("doc_id") or "UNKNOWN")

    def _get_window(
            self,
            splits: list[Document],
            i: int,
    ) -> list[Document]:
        """
        Returns a list of docs of length `window` when possible.
        Tries to place index `i` near the middle of the window.
        """
        n = len(splits)
        if n == 0:
            return []

        window = min(self.sliding_window, n)
        half = window // 2

        start = i - half
        end = start + window

        # Clamp to [0, n] while preserving window length when possible
        if start < 0:
            start = 0
            end = window
        if end > n:
            end = n
            start = n - window

        return splits[start:end]

    def _get_context_for_split(self, document_contents: list[Document], split: Document):
        separator = "\n\n"
        neighbour_content = separator.join(doc.page_content for doc in document_contents)

        prompt = self.prompt.format(
            doc_content=neighbour_content,
            chunk_content=split.page_content,
        )

        result = self.language_model.invoke(prompt)
        context = getattr(result, "content", str(result))

        return context
