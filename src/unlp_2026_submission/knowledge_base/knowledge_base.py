from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.response_synthesizers import (
    get_response_synthesizer,
    ResponseMode,
    BaseSynthesizer
)
from llama_index.core.postprocessor import LongContextReorder, SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.embeddings.langchain import LangchainEmbedding

from unlp_2026_submission.entities import DocumentPage
from unlp_2026_submission.knowledge_base.storage import (
    SimpleDocumentStore,
    StorageContext,
    SimpleIndexStore,
    SimpleGraphStore,
    QdrantVectorStore,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.response import Response
from typing import List
from pathlib import Path
from qdrant_client.http.models import FieldCondition, MatchValue, Range

from unlp_2026_submission.embeddings import EmbeddingsModel
from unlp_2026_submission.language_models import LlamaLanguageModel
from unlp_2026_submission.config import KnowledgeBaseConfig


STORAGE_CONTEXT_REQUIRED_FILES = {
    "docstore.json",
    "index_store.json",
}

def is_persisted_storage_context_exist(persist_dir: str) -> bool:
    p = Path(persist_dir)
    if not p.exists() or not p.is_dir():
        return False

    existing = {f.name for f in p.iterdir() if f.is_file()}
    return STORAGE_CONTEXT_REQUIRED_FILES.issubset(existing)

class KnowledgeBase:
    _storage_context: StorageContext
    _vector_store_index: VectorStoreIndex

    _language_model: LlamaLanguageModel
    _embeddings_model: EmbeddingsModel
    _config: KnowledgeBaseConfig

    _node_postprocessors: List[BaseNodePostprocessor]
    _response_synthesizer: BaseSynthesizer

    def __init__(
            self,
            storage_context: StorageContext,
            vector_store_index: VectorStoreIndex,
            language_model: LlamaLanguageModel,
            embeddings_model: EmbeddingsModel,
            config: KnowledgeBaseConfig,
    ):
        self._storage_context = storage_context
        self._vector_store_index = vector_store_index
        self._language_model = language_model
        self._embeddings_model = embeddings_model
        self._config = config

        # TODO add more postprocessors
        self._node_postprocessors = [LongContextReorder()]
        # TODO think about response synthesizer
        self._response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            llm=self._language_model,
            # service_context=service_context,
            # text_qa_template=text_qa_template,
            # refine_template=refine_template,
            # use_async=False,
            # streaming=False,
        )

    @property
    def storage_context(self):
        return self._storage_context

    @property
    def vector_store_index(self):
        return self._vector_store_index

    @property
    def vector_store(self):
        return self._storage_context.vector_store

    @property
    def doc_store(self):
        return self._storage_context.docstore

    @property
    def index_store(self):
        return self._storage_context.index_store

    @property
    def graph_store(self):
        return self._storage_context.graph_store

    @property
    def node_postprocessors(self):
        return self._node_postprocessors

    @node_postprocessors.setter
    def node_postprocessors(self, value: List[BaseNodePostprocessor]):
        self._node_postprocessors = value

    @property
    def response_synthesizer(self):
        return self._response_synthesizer

    @response_synthesizer.setter
    def response_synthesizer(self, value: BaseSynthesizer):
        self._response_synthesizer = value

    def retrieve_page(
            self,
            search_query: str,
            filters: Filter | None = None,
            hybrid_top_k: int = 2,
            similarity_top_k: int = 2,
            sparse_top_k: int = 2,
    ) -> DocumentPage | None:
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.5)

        nodes = self.retrieve(
            search_query=search_query,
            filters=filters,
            hybrid_top_k=hybrid_top_k,
            similarity_top_k=similarity_top_k,
            sparse_top_k=sparse_top_k,
        )

        postprocessed_nodes = postprocessor.postprocess_nodes(
            nodes=nodes
        )

        if not postprocessed_nodes:
            return None

        most_relevant_node = postprocessed_nodes[0]
        # TODO: page can be chunked for more than one chunk, need to handle this
        return DocumentPage.get_from_node_with_sore(most_relevant_node)

    def query(
            self,
            search_query: str,
            filters: Filter | None = None,
            hybrid_top_k: int = 2,
            similarity_top_k: int = 2,
            sparse_top_k: int = 2,
            response_synthesizer: BaseSynthesizer | None = None,
    ) -> Response:
        query_engine = self.vector_store_index.as_query_engine(
            response_synthesizer=(
                response_synthesizer
                if response_synthesizer is not None
                else self.response_synthesizer
            ),
            node_postprocessors=self.node_postprocessors,
            llm=self._language_model,
            embeddings_model=self._embeddings_model,
            vector_store_kwargs={"qdrant_filters": filters},
            hybrid_top_k=hybrid_top_k,
            similarity_top_k=similarity_top_k,
            sparse_top_k=sparse_top_k
        )

        return query_engine.query(search_query)

    def retrieve(
            self,
            search_query: str,
            filters: Filter | None = None,
            hybrid_top_k: int = 2,
            similarity_top_k: int = 2,
            sparse_top_k: int = 2,
    ) -> list[NodeWithScore]:
        retriever = self.vector_store_index.as_retriever(
            vector_store_kwargs={"qdrant_filters": filters},
            hybrid_top_k=hybrid_top_k,
            similarity_top_k=similarity_top_k,
            sparse_top_k=sparse_top_k
        )

        return retriever.retrieve(search_query)

    @classmethod
    def create_empty(
            cls,
            language_model: LlamaLanguageModel,
            embeddings_model: EmbeddingsModel,
            config: KnowledgeBaseConfig,
            should_persist: bool = True,
    ):
        if should_persist:
            Path(config.kb_store_root_dir).mkdir(exist_ok=True)

            qdrant_client = QdrantClient(path=config.vector_store_path)
        else:
            qdrant_client = QdrantClient(location=":memory:")


        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=config.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
        )

        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore(),
            index_store=SimpleIndexStore(),
            vector_store=vector_store,
            graph_store=SimpleGraphStore(),
        )

        vector_store_index = VectorStoreIndex(
            nodes=[],
            embed_model=LangchainEmbedding(embeddings_model),
            storage_context=storage_context
        )

        knowledge_base = cls(
            language_model=language_model,
            storage_context=storage_context,
            vector_store_index=vector_store_index,
            config=config,
            embeddings_model=embeddings_model
        )

        if should_persist:
            vector_store_index.storage_context.persist(
                persist_dir=config.context_path
            )

        return knowledge_base

    @classmethod
    def load(
            cls,
            language_model: LlamaLanguageModel,
            embeddings_model: EmbeddingsModel,
            config: KnowledgeBaseConfig,
            should_persist: bool = True,
    ):
        if not is_persisted_storage_context_exist(config.context_path):
            return KnowledgeBase.create_empty(
                language_model=language_model,
                embeddings_model=embeddings_model,
                config=config,
                should_persist=should_persist,
            )

        qdrant_client = QdrantClient(path=config.vector_store_path)

        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=config.collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
        )

        storage_context = StorageContext.from_defaults(
            persist_dir=config.context_path,
            vector_store=vector_store
        )

        vector_store_index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=LangchainEmbedding(embeddings_model),
        )

        knowledge_base = cls(
            language_model=language_model,
            storage_context=storage_context,
            vector_store_index=vector_store_index,
            config=config,
            embeddings_model=embeddings_model,
        )

        return knowledge_base
