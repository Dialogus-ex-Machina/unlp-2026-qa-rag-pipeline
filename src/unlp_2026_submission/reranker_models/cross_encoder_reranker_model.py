from typing import Dict, Any
from sentence_transformers import CrossEncoder

from unlp_2026_submission.reranker_models import RerankerModel
from unlp_2026_submission.entities import RelevantDocument


class CrossEncoderRerankerModel(RerankerModel):
    model: CrossEncoder

    def __init__(
            self,
            model_name: str,
            cache_dir: str,
            trust_remote_code: bool = True,
            **model_kwargs: Dict[str, Any]
    ):
        self.model = CrossEncoder(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            **model_kwargs
        )

    def rerank(
            self,
            query: str,
            documents: list[RelevantDocument]
    ) -> list[RelevantDocument]:
        document_texts = [doc.text for doc in documents]

        rerank_result = self.model.rank(
            query=query,
            documents=document_texts
        )

        reranked_documents = []

        for result in rerank_result :
            document = documents[result["corpus_id"]]
            document.relevance_score = result["score"]
            reranked_documents.append(document)

        results = sorted(reranked_documents, key=lambda x: x.relevance_score, reverse=True)
        return results