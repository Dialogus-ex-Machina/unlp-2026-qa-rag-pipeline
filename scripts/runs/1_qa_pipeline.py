import random

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow.nodes import (
    SimpleDocumentsRetrievalNode,
    SimpleQuestionAnswerNode,
    TopKRelevantDocumentAugmentation,
    MockDomainRoutingNode,
)
from unlp_2026_submission.workflow.nodes.logprob_reranker_model_node import LogprobRerankerModelNode
from unlp_2026_submission.workflow.prompts import UkrQAPrompt
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder


def main():
    config = Config(
        language_model_name="Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q8_0.gguf",
        embeddings_model_name="bflhc/Octen-Embedding-0.6B",
    )

    language_model = LanguageModelFactory.create(
        config=config,
    ).get_language_model()
    embeddings_model = EmbeddingsModelFactory.create(config).get_embeddings_model()

    qa_prompt = UkrQAPrompt()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        **config.vector_store
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
        )
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
        dataset_name=AccuracyDatasetName.FULL
    ).get_dataset()
    question = random.Random().choice(dataset)

    response = workflow.invoke(
        input={ 'question': question }
    )

    print('Response:', response)

if __name__ == "__main__":
    main()
