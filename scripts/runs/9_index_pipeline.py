from pathlib import Path

from unlp_2026_submission.rag.index import IndexState, IndexRunner
from unlp_2026_submission.rag.index.nodes import (
    DoclingMarkdownPageSplitNode,
    DoclingConverterLoadNode,
    HybridEmbedStoreNode,
)
from merlin.models.embeddings import EmbeddingsFactory, EmbeddingsSpec


def get_pdf_filepaths(documents_dir: str = "../documents") -> list[str]:
    p = Path(documents_dir)
    # recursive; use p.glob("*.pdf") if you only want top-level
    return sorted(str(fp) for fp in p.rglob("*.pdf") if fp.is_file())


spec = EmbeddingsSpec(
    provider="huggingface",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
embeddings = EmbeddingsFactory.create_all_embeddings_factory().create(spec)

"""Index pipeline configuration:
1) Convert: DoclingConverterLoadNode (PDF -> Markdown with page markers)
2) Split: DoclingMarkdownPageSplitNode (Markdown -> page-level Documents)
3) Embed+Store: HybridEmbedStoreNode (dense + sparse in Qdrant)
"""
nodes = [
    DoclingConverterLoadNode(device="cuda"),
    DoclingMarkdownPageSplitNode(),
    HybridEmbedStoreNode(embeddings),
]

index_runner = IndexRunner(nodes)

filepaths = get_pdf_filepaths("../documents/dummy")

initial_state: IndexState = {
    # "filepaths": ["../documents/test.pdf"],
    "filepaths": filepaths,
    "vector_store_path": "../vector_dbs/qdrant_db_9_minilm_384",
}

final_state = index_runner.run(initial_state)

# TODO: ADD RAG Runner
