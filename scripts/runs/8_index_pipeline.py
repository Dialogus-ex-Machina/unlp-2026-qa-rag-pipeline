from pathlib import Path

from merlin.rag.index import IndexState, IndexRunner
from merlin.rag.index.nodes import DoclingLoadSplitNode, EmbedStoreNode
from merlin.models.embeddings import EmbeddingsFactory, EmbeddingsSpec

def get_pdf_filepaths(documents_dir: str = "../documents") -> list[str]:
    p = Path(documents_dir)
    # recursive; use p.glob("*.pdf") if you only want top-level
    return sorted(str(fp) for fp in p.rglob("*.pdf") if fp.is_file())

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
    DoclingLoadSplitNode(embeddings, device="cuda", max_tokens=256),
    EmbedStoreNode(embeddings)
]

index_runner = IndexRunner(nodes)

filepaths = get_pdf_filepaths("../documents/pdf")

initial_state: IndexState = {
    # "filepaths": ["../documents/test.pdf"],
    "filepaths": filepaths,
    "vector_store_path": "../vector_dbs/qdrant_db_8"
}

final_state = index_runner.run(initial_state)

# TODO: ADD RAG Runner
