import random
from pathlib import Path

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow.nodes import (
    MockDomainRoutingNode,
    SimpleDocumentsRetrievalNode,
    SimpleQuestionAnswerNode,
    TopKRelevantDocumentAugmentation,
)
from unlp_2026_submission.workflow.nodes.logprob_reranker_model_node import (
    LogprobRerankerModelNode,
)
from unlp_2026_submission.workflow.prompts import UkrQAPrompt
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_root_dir = repo_root / "src" / "data"
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

    domain_pipeline_nodes = [
        SimpleDocumentsRetrievalNode(
            vector_store=vector_store,
        ),
        LogprobRerankerModelNode(
            language_model=language_model,
        ),
        TopKRelevantDocumentAugmentation(
            top_k=4,
        ),
        SimpleQuestionAnswerNode(
            language_model=language_model,
            prompt=qa_prompt,
        ),
    ]
    workflow = (
        QAWorkflowBuilder.create()
        .add_domain_routing_node(
            MockDomainRoutingNode()
        )
        .add_sport_domain_nodes(domain_pipeline_nodes)
        .add_medicine_domain_nodes(domain_pipeline_nodes)
        .add_other_domain_nodes(domain_pipeline_nodes)
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
