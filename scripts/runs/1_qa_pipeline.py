import random

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import SentenceTransformerEmbeddingModel
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName
from unlp_2026_submission.language_models import LlamaCppLanguageModel
from unlp_2026_submission.workflow.nodes import (
    MostRelevantDocumentAugmentationNode,
    SimpleDocumentsRetrievalNode,
    SimpleQuestionAnswerNode,
    LLMDomainRoutingNode,
)
from unlp_2026_submission.workflow.prompts import ENDomainClassificationPrompt
from unlp_2026_submission.workflow.prompts.qa_prompt import QAPrompt
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder


def main():
    config = Config(
        language_model_name="Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q8_0.gguf",
        embeddings_model_name="bflhc/Octen-Embedding-0.6B",
    )

    language_model = LlamaCppLanguageModel.create(
        config=config,
    )
    embeddings_model = SentenceTransformerEmbeddingModel.create(config)

    qa_prompt = QAPrompt()
    domain_classification_prompt = ENDomainClassificationPrompt()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        **config.vector_store,
    )

    domain_pipeline_nodes = [
        SimpleDocumentsRetrievalNode(
            vector_store=vector_store,
        ),
        MostRelevantDocumentAugmentationNode(),
        SimpleQuestionAnswerNode(
            language_model=language_model,
            prompt=qa_prompt,
        )
    ]
    workflow = (
        QAWorkflowBuilder.create()
        .add_domain_routing_node(
            LLMDomainRoutingNode(
                language_model=language_model,
                prompt=domain_classification_prompt,
            )
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
