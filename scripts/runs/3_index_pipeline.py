from langchain_text_splitters import RecursiveCharacterTextSplitter

from merlin.rag.index import IndexState, IndexRunner
from merlin.rag.index.nodes import EmbedStoreNode, SplitNode, DoclingPageLoadNode
from merlin.models.embeddings import EmbeddingsFactory, EmbeddingsSpec


spec = EmbeddingsSpec(
    provider="huggingface",
    model_name="bflhc/Octen-Embedding-0.6B",
)
embeddings = EmbeddingsFactory.create_all_embeddings_factory().create(spec)

"""Index pipeline configuration:
1) Load: DoclingPageLoad(Octen-Embedding-0.6B)
2) Split: RecursiveCharacterTextSplitter(size=400, overlap=0, start_index=true)
3) Embed+Store: Qdrand(Octen-Embedding-0.6B)
"""
nodes = [
    DoclingPageLoadNode(embeddings),
    SplitNode(
        RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            add_start_index=True,
        )
    ),
    EmbedStoreNode(embeddings)
]

index_runner = IndexRunner(nodes)

initial_state: IndexState = {
    "filepaths": ["../documents/test.pdf"],
    "vector_store_path": "../vector_dbs/qdrant_db"
}

final_state = index_runner.run(initial_state)

# TODO: ADD RAG Runner
