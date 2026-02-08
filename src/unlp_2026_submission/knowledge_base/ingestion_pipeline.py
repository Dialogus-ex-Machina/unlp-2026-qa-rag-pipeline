from typing import List, Optional, Sequence

from langchain_experimental.text_splitter import SemanticChunker
from llama_index.core.constants import DEFAULT_PIPELINE_NAME, DEFAULT_PROJECT_NAME
from llama_index.core.schema import TransformComponent
from llama_index.core.ingestion import IngestionPipeline as BaseIngestionPipeline
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.core.storage.docstore import BaseDocumentStore
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core import Document
from llama_index.core.node_parser import LangchainNodeParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from unlp_2026_submission.chunking_evaluation.chunking import ClusterSemanticChunker

from unlp_2026_submission.config import KnowledgeBaseConfig
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
            # LangchainNodeParser(
                # ClusterSemanticChunker(
                #     max_chunk_size=1600,
                #     min_chunk_size=500,
                #     embedding_function=embedding_function, batch_size=10
                # )
                # RecursiveCharacterTextSplitter(
                #     chunk_size=800,
                #     chunk_overlap=0,
                #     add_start_index=True,
                # ),
                # SemanticChunker(embeddings=embeddings_model, add_start_index=True),
            # ),
            LangchainEmbedding(langchain_embeddings=embeddings_model, embed_batch_size=1),
        ]

        ingestion_pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store,
            docstore=doc_store,
        )

        return ingestion_pipeline
