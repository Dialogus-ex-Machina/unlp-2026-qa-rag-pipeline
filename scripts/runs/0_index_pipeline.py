from merlin.rag.index import IndexState, IndexRunner
from merlin.rag.index.nodes import (
    EmbedStoreNode,
    DelimitedPageLoadNode,
    ProxySplitNode,
    ContextualSplitsAugmentationNode
)
from unlp_2026_submission.config import Config
from unlp_2026_submission.models.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.models.language_models import LlamaCppLanguageModel

config = Config(
    embeddings_model_name="bflhc/Octen-Embedding-0.6B",
    language_model_name="unsloth/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q2_K.gguf"
)
embeddings_model = EmbeddingsModelFactory.create(config).get_embeddings_model()
language_model = LlamaCppLanguageModel.create(config)

"""Index pipeline configuration:
1) Load: DelimitedPageLoadNode
2) Split: ContextualSplitsAugmentationNode via llm
3) Embed+Store: Qdrand(Octen-Embedding-0.6B)
"""

def on_success():
    """
    Free memory before embeddings part
    """
    language_model.client.close()

nodes = [
    DelimitedPageLoadNode(),
    ProxySplitNode(),
    ContextualSplitsAugmentationNode(
        language_model=language_model,
        on_success=on_success,
    ),
    EmbedStoreNode(
        embeddings=embeddings_model,
        batch_size=10,
    )
]

index_runner = IndexRunner(nodes)

initial_state: IndexState = {
    "filepaths": ["../documents/test.pdf"],
    "vector_store_path": config.vector_store['path'],
}

final_state = index_runner.run(initial_state)

# TODO: ADD RAG Runner
