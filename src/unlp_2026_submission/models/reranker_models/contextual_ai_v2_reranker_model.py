import torch
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from unlp_2026_submission.entities import RelevantDocument

from .reranker_model import RerankerModel


def _format_prompt(query: str, instruction: str, doc: str) -> str:
    """Format query and document into a prompt for reranking."""
    if instruction:
        instruction = f" {instruction}"
    return f"Check whether a given document contains information helpful to answer the query.\n<Document> {doc}\n<Query> {query}{instruction} ??"


class ContextualAIV2RerankerModel(RerankerModel):
    def __init__(
            self,
            cache_dir: str,
            model_name: str = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b",
            device: str = "cuda",
            batch_size: int = 4,
            instruction: Optional[str] = None,
            max_length: int = 2048,
            **model_kwargs: Dict[str, Any],
    ):
        self._batch_size = batch_size
        self._max_length = max_length
        self._instruction = instruction or "Prioritize documents that helpful to answer the query."

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=True,
            **model_kwargs,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            device_map=device,
            **model_kwargs,
        ).eval()

    def rerank(
            self,
            query: str,
            documents: List[RelevantDocument],
    ) -> List[RelevantDocument]:
        if not documents:
            return []

        prompts = [
            _format_prompt(query, self._instruction, doc.text)
            for doc in documents
        ]

        all_scores: List[float] = []
        for start in range(0, len(prompts), self._batch_size):
            batch_prompts = prompts[start:start + self._batch_size]
            batch_scores = self._compute_batch_scores(batch_prompts)
            all_scores.extend(batch_scores)

        results: List[RelevantDocument] = []
        for i, score in enumerate(all_scores):
            document = documents[i]
            document.relevance_score = score
            results.append(document)

        results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        return results

    @torch.no_grad()
    def _compute_batch_scores(self, prompts: List[str]) -> List[float]:
        """Compute relevance scores for a batch of prompts."""
        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        input_ids = enc["input_ids"].to(self._model.device)
        attention_mask = enc["attention_mask"].to(self._model.device)

        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        next_logits = out.logits[:, -1, :]  # [batch, vocab]
        scores = next_logits[:, 0].float().tolist()
        return scores
