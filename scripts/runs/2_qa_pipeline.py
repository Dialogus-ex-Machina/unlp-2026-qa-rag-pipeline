import random
from pathlib import Path

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from unlp_2026_submission.config import Config
from unlp_2026_submission.models.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName
from unlp_2026_submission.models.language_models import LanguageModelFactory
from unlp_2026_submission.rag.qa.nodes import (
    SimpleRetrievalNode,
    SimpleQuestionAnswerNode,
    TopKDocsContextCreationNode,
    LogprobRerankerNode,
)
from unlp_2026_submission.rag.qa.prompts import UkrQAPrompt
from unlp_2026_submission.rag.qa.qa_workflow_builder import QAWorkflowBuilder


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_root_dir = repo_root / "data"
    vector_store_path = repo_root / "scripts" / "vector_dbs" / "qdrant_db_9_minilm_384"

    config = Config(
        language_model_name="Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q8_0.gguf",
        embeddings_model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    config.data_root_dir = str(data_root_dir)
    config.vector_store["path"] = str(vector_store_path)

    language_model = LanguageModelFactory.create(
        config=config,
    ).get_language_model()
    embeddings_model = EmbeddingsModelFactory.create(config).get_embeddings_model()

    qa_prompt = UkrQAPrompt()

    vector_store = QdrantVectorStore(
        client=QdrantClient(path=config.vector_store["path"]),
        collection_name=config.vector_store["collection_name"],
        embedding=embeddings_model,
        sparse_embedding=FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

    nodes = [
        SimpleRetrievalNode(
            vector_store=vector_store,
        ),
        LogprobRerankerNode(
            language_model=language_model,
        ),
        TopKDocsContextCreationNode(
            top_k=4,
        ),
        SimpleQuestionAnswerNode(
            language_model=language_model,
            prompt=qa_prompt,
        ),
    ]
    workflow = (
        QAWorkflowBuilder.create()
        .add_nodes(nodes)
        .build()
    )

    dataset = AccuracyDatasetFactory.create(
        data_root_dir=config.data_root_dir,
        dataset_name=AccuracyDatasetName.FULL,
    ).get_dataset()
    question = random.Random().choice(dataset)

    response = workflow.invoke(
        input={"question": question}
    )

    print("Response:", response)


if __name__ == "__main__":
    main()
