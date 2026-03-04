from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from merlin.rag.index import IndexState, IndexRunner
from merlin.rag.index.nodes import EmbedStoreNode, SimpleSplitNode, DoclingPageLoadNode
from merlin.models.embeddings import EmbeddingsFactory, EmbeddingsSpec

def get_pdf_filepaths(documents_dir: str = "../documents") -> list[str]:
    p = Path(documents_dir)
    # recursive; use p.glob("*.pdf") if you only want top-level
    return sorted(str(fp) for fp in p.rglob("*.pdf") if fp.is_file())

spec = EmbeddingsSpec(
    provider="huggingface",
    model_name="bflhc/Octen-Embedding-0.6B",
    model_kwargs={"device":"cuda"},
)
embeddings = EmbeddingsFactory.create_all_embeddings_factory().create(spec)

"""Index pipeline configuration:
1) Load: DoclingPageLoad(Octen-Embedding-0.6B)
2) Split: Skipped
3) Embed+Store: Qdrand(Octen-Embedding-0.6B)
"""
nodes = [
    DoclingPageLoadNode(embeddings, device="cuda"),
    SimpleSplitNode(
        RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=0,
            add_start_index=True,
        )
    ),
    EmbedStoreNode(embeddings)
]

index_runner = IndexRunner(nodes)

filepaths = get_pdf_filepaths("../documents/pdf")

initial_state: IndexState = {
    # "filepaths": ["../documents/test.pdf"],
    "filepaths": filepaths,
    "vector_store_path": "../vector_dbs/qdrant_db_5"
}

final_state = index_runner.run(initial_state)

# TODO: ADD RAG Runner
