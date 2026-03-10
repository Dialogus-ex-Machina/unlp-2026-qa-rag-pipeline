from collections import defaultdict
from typing import Optional, Callable, Iterator

from langchain_core.documents import Document

from unlp_2026_submission.rag.index.index_state import IndexState
from unlp_2026_submission.models.language_models import LanguageModel
from unlp_2026_submission.rag.qa.prompts import SplitsAugmentationPrompt, UkrContextualSplitsAugmentationPrompt


class ContextualSplitsAugmentationNode:
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

        groups: dict[str, list[int]] = defaultdict(list)
        for idx, d in enumerate(splits):
            groups[self.get_doc_key(d)].append(idx)

        for _, idxs in groups.items():
            group_docs = [splits[j] for j in idxs]

            for window_docs in self._iter_windows(group_docs):
                context = self._get_context_for_window(window_docs)
                # print('Added context:', context)

                for split in window_docs:
                    if context is not None and "context" not in split.metadata:
                        split.metadata["context"] = context

        for split in splits:
            context = split.metadata.pop("context", None)
            if context:
                split.page_content = f"{split.page_content}\n\n{context}"

        if self.on_success:
            self.on_success()

        return {"splits": splits}

    def get_doc_key(self, d: Document) -> str:
        return str(d.metadata.get("source") or d.metadata.get("file_name") or d.metadata.get("doc_id") or "UNKNOWN")

    def _iter_windows(self, group_docs: list[Document]) -> Iterator[list[Document]]:
        """Генератор, який розбиває список чанків на батчі (вікна)."""
        n = len(group_docs)
        if n == 0:
            return

        window_size = min(self.sliding_window, n)

        for start_idx in range(0, n, window_size):
            end_idx = min(start_idx + window_size, n)

            if end_idx - start_idx < min(window_size, 2) and start_idx > 0:
                start_idx = max(0, end_idx - min(window_size, 2))

            yield group_docs[start_idx:end_idx]

    def _get_context_for_window(self, window_docs: list[Document]) -> str:
        prompt_text = self.prompt.format(document_splits=window_docs)

        try:
            result = self.language_model.invoke(prompt_text)
            context = getattr(result, "content", str(result))
            return context
        except Exception as e:
            print(f"Error in ContextualSplitsAugmentationNode: {e}")
            return ""