from merlin.rag.index import IndexState, IndexRunner
from merlin.rag.index.nodes import DoclingLoadSplitNode, EmbedStoreNode
from merlin.models.embeddings import EmbeddingsFactory, EmbeddingsSpec

spec = EmbeddingsSpec(
    provider="huggingface",
    model_name="bflhc/Octen-Embedding-0.6B",
)
embeddings = EmbeddingsFactory.create_all_embeddings_factory().create(spec)

"""Index pipeline configuration:
1) Load+Split: DoclingLoadSplit(Octen-Embedding-0.6B)
2) Embed+Store: Qdrand(Octen-Embedding-0.6B)
"""
nodes = [
    DoclingLoadSplitNode(embeddings, device="cpu"),
    EmbedStoreNode(embeddings)
]

index_runner = IndexRunner(nodes)

initial_state: IndexState = {
    "filepaths": ["../documents/test.pdf"],
    "vector_store_path": "../vector_dbs/qdrant_db"
}

final_state = index_runner.run(initial_state)

# TODO: ADD RAG Runner
