import torch
from typing import Any, Dict, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from unlp_2026_submission.entities import RelevantDocument

from .reranker_model import RerankerModel


class BGEV2RerankerModel(RerankerModel):
    def __init__(
            self,
            cache_dir: str,
            model_name: str = "BAAI/bge-reranker-v2-m3",
            device: str = "cuda",
            batch_size: int = 2,
            max_length: int = 2048,
            **model_kwargs: Dict[str, Any],
    ):
        self._batch_size = batch_size
        self._max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        )
        self._model.to(device)
        self._model.eval()

    def rerank(
            self,
            query: str,
            documents: List[RelevantDocument],
    ) -> List[RelevantDocument]:
        if not documents:
            return []

        all_scores: List[float] = []
        for start in range(0, len(documents), self._batch_size):
            batch_docs = documents[start:start + self._batch_size]
            batch_scores = self._compute_batch_scores(query, batch_docs)
            all_scores.extend(batch_scores)

        results: List[RelevantDocument] = []
        for i, score in enumerate(all_scores):
            document = documents[i]
            document.relevance_score = score
            results.append(document)

        results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        return results

    @torch.no_grad()
    def _compute_batch_scores(
            self,
            query: str,
            documents: List[RelevantDocument],
    ) -> List[float]:
        """Compute relevance scores for a batch of (query, document) pairs."""
        pairs = [[query, doc.text] for doc in documents]
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._max_length,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        logits = self._model(**inputs, return_dict=True).logits.view(-1).float()
        return logits.tolist()
