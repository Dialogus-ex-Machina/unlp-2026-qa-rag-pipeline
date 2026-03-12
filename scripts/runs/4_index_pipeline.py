from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from unlp_2026_submission.config import Config
from unlp_2026_submission.models.embeddings import SentenceTransformerEmbeddingModel
from unlp_2026_submission.rag.index import IndexState, IndexRunner
from unlp_2026_submission.rag.index.nodes import EmbedStoreNode, PyMuPDFLoadNode, SimpleSplitNode

def get_pdf_filepaths(documents_dir: str = "../documents") -> list[str]:
    p = Path(documents_dir)
    # recursive; use p.glob("*.pdf") if you only want top-level
    return sorted(str(fp) for fp in p.rglob("*.pdf") if fp.is_file())

config = Config(
    embeddings_model_name="bflhc/Octen-Embedding-0.6B",
)

embeddings = SentenceTransformerEmbeddingModel(
    model_name_or_path=config.embeddings_model_name,
    cache_folder=config.downloaded_models_cache_dir,
)

vector_store_client = QdrantClient(
    path=config.vector_store['path']
)

"""Index pipeline configuration:
1) Load: PyPDFLoader
2) Split: RecursiveCharacterTextSplitter(size=400, overlap=0, start_index=true)
3) Embed+Store: Qdrand(Octen-Embedding-0.6B)
"""
nodes = [
    PyMuPDFLoadNode(),
    SimpleSplitNode(
        RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            add_start_index=True,
        )
    ),
    EmbedStoreNode(
        vector_store_client=vector_store_client,
        embeddings=embeddings,
    )
]

index_runner = IndexRunner(nodes)

filepaths = get_pdf_filepaths("../documents/pdf")

initial_state: IndexState = {
    # "filepaths": ["../documents/test.pdf"],
    "filepaths": filepaths,
    "vector_store_path": "../vector_dbs/qdrant_db_4"
}

final_state = index_runner.run(initial_state)

# TODO: ADD RAG Runner
