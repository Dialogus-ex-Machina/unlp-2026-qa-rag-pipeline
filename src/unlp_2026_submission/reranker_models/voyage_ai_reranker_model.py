from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

import voyageai
from langchain_core.utils import convert_to_secret_str
from pydantic import ConfigDict, SecretStr, model_validator, BaseModel

from unlp_2026_submission.entities import RelevantDocument


class VoyageAIRerankerModel(BaseModel):
    client: voyageai.Client = None
    """VoyageAI clients to use for compressing documents."""
    voyage_api_key: Optional[SecretStr] = None
    """VoyageAI API key. Must be specified directly or via environment variable
        VOYAGE_API_KEY."""
    base_url: Optional[str] = None
    """Custom API endpoint URL. If not provided, the VoyageAI SDK determines
    the default based on the API key."""
    model: str = 'rerank-2.5'
    """Model to use for reranking."""
    top_k: Optional[int] = None
    """Number of documents to return."""
    truncation: bool = True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        voyage_api_key = values.get("voyage_api_key") or os.getenv(
            "VOYAGE_API_KEY", None
        )
        if voyage_api_key:
            api_key_secretstr = convert_to_secret_str(voyage_api_key)
            values["voyage_api_key"] = api_key_secretstr

            api_key_str = api_key_secretstr.get_secret_value()
        else:
            api_key_str = None

        base_url = values.get("base_url")
        values["client"] = voyageai.Client(api_key=api_key_str, base_url=base_url)

        return values


    def rerank(
        self,
        query: str,
        documents: List[RelevantDocument],
    ) -> List[RelevantDocument]:
        if len(documents) == 0:
            return []

        doc_texts = [doc.text for doc in documents]
        rerank_result = self.client.rerank(
            query=query,
            documents=doc_texts,
            model=self.model,
            top_k=self.top_k,
            truncation=self.truncation,
        )

        reranked_documents = []
        for res in rerank_result.results:
            document = documents[res.index]
            document.relevance_score = res.relevance_score
            reranked_documents.append(document)
        return reranked_documents