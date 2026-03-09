from collections import defaultdict
from typing import Optional, Callable

from langchain_core.documents import Document

from unlp_2026_submission.rag.index.index_state import IndexState
from unlp_2026_submission.models.language_models import LanguageModel
from unlp_2026_submission.rag.qa.prompts import UkrContextualSplitsAugmentationPrompt, SplitsAugmentationPrompt


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
                split.metadata["context"] = context
                # print('Added context:', context)

        for split in splits:
            context = split.metadata.get("context", None)

            if context is not None:
                split.page_content = f"{split.page_content}\n\n{context}"
                del split.metadata["context"]

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
        Returns documents in the static window containing index i.
        Documents are cut into non-overlapping windows of size window_size.
        Window for index i starts at (i // window_size) * window_size.
        If the last window would have only 1 split, it is expanded backward
        to include at least 2 splits.
        """
        n = len(splits)
        if n == 0:
            return []

        window_size = min(self.sliding_window, n)
        window_start = (i // window_size) * window_size
        window_end = min(window_start + window_size, n)

        # Avoid last window with only 1 split: expand backward
        if window_end - window_start < min(window_size, 2) and window_start > 0:
            window_start = max(0, window_end - min(window_size, 2))

        return splits[window_start:window_end]

    def _get_context_for_split(
            self,
            document_splits: list[Document],
            target_split: Document
    ):
        prompt = self.prompt.format(
            document_splits=document_splits,
            target_split=target_split,
        )

        try:
            result = self.language_model.invoke(prompt)
            context = getattr(result, "content", str(result))

            return context
        except Exception as e:
            print(f"Error in ContextualSplitsAugmentationNode: {e}")
            return ""
