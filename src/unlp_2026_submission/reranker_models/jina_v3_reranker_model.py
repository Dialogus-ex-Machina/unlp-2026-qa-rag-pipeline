from typing import Dict, Any, List
from transformers import AutoModel

from unlp_2026_submission.reranker_models import RerankerModel
from unlp_2026_submission.entities import RelevantDocument


class JinaV3RerankerModel(RerankerModel):
    model: Any

    def __init__(
            self,
            cache_dir: str,
            device: str = "cuda",
            batch_size: int = 4,
            **model_kwargs: Dict[str, Any]
    ):
        self._batch_size = batch_size
        self.model = AutoModel.from_pretrained(
            'jinaai/jina-reranker-v3',
            dtype="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            device_map=device,
            **model_kwargs
        )
        self.model.eval()

    def rerank(
            self,
            query: str,
            documents: list[RelevantDocument]
    ) -> list[RelevantDocument]:
        reranked_documents = []

        for start in range(0, len(documents), self._batch_size):
            batch_docs = documents[start:start + self._batch_size]

            document_texts = [doc.text for doc in batch_docs]

            batch_results = self.model.rerank(query, document_texts)

            for result in batch_results:
                document = batch_docs[result["index"]]
                document.relevance_score = result["relevance_score"]
                reranked_documents.append(document)

        results = sorted(reranked_documents, key=lambda x: x.relevance_score, reverse=True)
        return results
