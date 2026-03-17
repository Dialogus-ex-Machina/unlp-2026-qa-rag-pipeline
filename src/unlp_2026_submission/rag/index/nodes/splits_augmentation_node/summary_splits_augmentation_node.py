from collections import defaultdict
from typing import Dict, Any
import gc

from langchain_core.documents import Document

from unlp_2026_submission.models.language_models import LlamaCppLanguageModel, LanguageModel
from unlp_2026_submission.rag.index import IndexState
from unlp_2026_submission.rag.index.prompts import UkrSummarySplitsAugmentationPrompt


class SummarySplitsAugmentationNode:
    def __init__(
            self,
            language_model_kwargs: Dict[str, Any],
            char_length: int = 200,
            window_size: int = 6,
            prompt: UkrSummarySplitsAugmentationPrompt | None = None,
    ):
        self.char_length = char_length
        self.language_model_kwargs = language_model_kwargs
        self.window_size = max(0, int(window_size))
        self.prompt = prompt or UkrSummarySplitsAugmentationPrompt()

    def __call__(self, state) -> IndexState:
        splits: list[Document] = state["splits"]

        language_model = LlamaCppLanguageModel(
            **self.language_model_kwargs
        )

        # 1) group indices by doc key (preserve original order)
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, d in enumerate(splits):
            groups[self.get_doc_key(d)].append(idx)

        for _, idxs in groups.items():
            group_docs = [splits[j] for j in idxs]

            context = self._get_doc_summary(language_model, group_docs)
            print('Added context:', context)

            for global_idx in idxs:
                split = splits[global_idx]
                split.metadata["context"] = context

        for split in splits:
            context = split.metadata.get("context", None)

            if context is not None:
                split.page_content = f"{context}\n\n{split.page_content}"
                del split.metadata["context"]

        language_model.client.close()
        del language_model
        gc.collect()

        return {"splits": splits}

    def get_doc_key(self, d: Document) -> str:
        return str(d.metadata.get("source") or d.metadata.get("file_name") or d.metadata.get("doc_id") or "UNKNOWN")

    def _get_doc_summary(
            self,
            language_model: LanguageModel,
            document_splits: list[Document]
    ) -> str:
        prompt = self.prompt.format(
            document_splits=document_splits,
            char_length=self.char_length,
            window_size=self.window_size,
        )

        try:
            result = language_model.invoke(prompt)
            context = getattr(result, "content", str(result))

            return context
        except Exception as e:
            print(f"Error in SummarySplitsAugmentationNode: {e}")
            return ""

