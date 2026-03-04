from pathlib import Path

from unlp_2026_submission.rag.index import IndexState, IndexRunner
from unlp_2026_submission.rag.index.nodes import TxtLoadNode, HybridEmbedStoreNode, PageAwareDocumentSplitNode
from unlp_2026_submission.rag.index.splitters import SentenceSplitter
from unlp_2026_submission.config import Config
from unlp_2026_submission.models.embeddings import EmbeddingsModelFactory

def get_txt_filepaths(documents_dir: str = "../documents") -> list[str]:
    p = Path(documents_dir)
    # recursive; use p.glob("*.pdf") if you only want top-level
    return sorted(str(fp) for fp in p.rglob("*.txt") if fp.is_file())

"""Index pipeline configuration:
    1) Load: TxtLoadNode
    2) Split: SentenceSplitter(chunk_size=1200, chunk_overlap=0)
    3) HybridEmbedStoreNode()
"""

def main():
    config = Config()

    embeddings_model = EmbeddingsModelFactory.create(config).get_embeddings_model()

    nodes = [
        TxtLoadNode(),
        PageAwareDocumentSplitNode(
            splitter=SentenceSplitter(
                chunk_size=1200,
                chunk_overlap=0,
            )
        ),
        HybridEmbedStoreNode(
            embeddings=embeddings_model,
            batch_size=1,
            collection_name=config.vector_store['collection_name'],
        )
    ]

    index_runner = IndexRunner(nodes)

    filepaths = get_txt_filepaths(
        config.data_root_dir
    )

    initial_state: IndexState = {
        "filepaths": filepaths,
        "vector_store_path": config.vector_store['path'],
    }

    final_state = index_runner.run(initial_state)

    print('Final state:', final_state)

if __name__ == "__main__":
    main()