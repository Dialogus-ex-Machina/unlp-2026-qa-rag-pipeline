from typing import List, Optional, Sequence
from llama_index.core.constants import DEFAULT_PIPELINE_NAME, DEFAULT_PROJECT_NAME
from llama_index.core.schema import TransformComponent
from llama_index.core.ingestion import IngestionPipeline as BaseIngestionPipeline
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.embeddings.langchain import LangchainEmbedding

from unlp_2026_submission.config import KnowledgeBaseConfig
from unlp_2026_submission.knowledge_base.documents import Document
from unlp_2026_submission.knowledge_base.transformations import SentenceSplitter
from unlp_2026_submission.embeddings import EmbeddingsModel

class IngestionPipeline(BaseIngestionPipeline):
    def __init__(
            self,
            transformations: List[TransformComponent],
            name: Optional[str] | None = DEFAULT_PIPELINE_NAME,
            project_name: Optional[str] | None = DEFAULT_PROJECT_NAME,
            documents: Optional[Sequence[Document]] = None,
            vector_store: Optional[BasePydanticVectorStore] = None,
            # readers: Optional[List[ReaderConfig]] = None,
            # cache: Optional[IngestionCache] = None,
            docstore: Optional[BaseDocumentStore] = None,
            # docstore_strategy: DocstoreStrategy = DocstoreStrategy.UPSERTS,
            # disable_cache: bool = False,
        ):
        super().__init__(
            name=name,
            project_name=project_name,
            transformations=transformations,
            documents=documents,
            vector_store=vector_store,
            docstore=docstore,
        )

    @staticmethod
    def create(
            config: KnowledgeBaseConfig,
            embeddings_model: EmbeddingsModel,
            vector_store: Optional[BasePydanticVectorStore],
            doc_store: Optional[BaseDocumentStore] = None,
    ):
        transformations = [
            SentenceSplitter(),
            LangchainEmbedding(embeddings_model),
        ]

        ingestion_pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store,
            docstore=doc_store,
        )

        return ingestion_pipeline
