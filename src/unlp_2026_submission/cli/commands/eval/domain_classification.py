import asyncio

import typer
from typing import Annotated
import logging

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.evals.accuracy import (
    AccuracyDatasetFactory,
    AccuracyDatasetName,
)
from unlp_2026_submission.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.evals.create_experiment_name import create_experiment_name
from unlp_2026_submission.evals.domain_classification import evaluate_domain_classification
from unlp_2026_submission.workflow.nodes import (
    MostRelevantDocumentAugmentationNode,
    SimpleDocumentsRetrievalNode,
    LLMDomainRoutingNode,
    FakeQuestionAnswerNode,
)
from unlp_2026_submission.workflow.prompts import PromptsFactory
from unlp_2026_submission.workflow.prompts.domain_classification_prompt_type import DomainClassificationPromptType
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import LanguageModelFactory

app = typer.Typer()


@app.command('domain-classification')
def evaluate_domain_classification_command(
        dataset_name: Annotated[
            AccuracyDatasetName,
            typer.Option("--dataset", "-ds")
        ] = AccuracyDatasetName.FULL,
        domain_classification_prompt_type: Annotated[
            DomainClassificationPromptType,
            typer.Option("--classify-prompt")
        ] = DomainClassificationPromptType.SIMPLE_EN,
        language_model_name: Annotated[str, typer.Option("--model", "-m")] = None,
        model_provider_api_key: Annotated[str, typer.Option("--api-key", "-key")] = None,
        embeddings_model_name: Annotated[str, typer.Option("--embeddings-model", "-em")] = None,
        logging_level: Annotated[int, typer.Option("--logs", "-l")] = logging.INFO,
):
    """
        Evaluate domain classification
    """
    logging.basicConfig(level=logging_level)

    asyncio.run(
        _evaluate(
            dataset_name=dataset_name,
            domain_classification_prompt_type=domain_classification_prompt_type,
            language_model_name=language_model_name,
            model_provider_api_key=model_provider_api_key,
            embeddings_model_name=embeddings_model_name,
        )
    )


async def _evaluate(
        dataset_name: AccuracyDatasetName,
        domain_classification_prompt_type: DomainClassificationPromptType,
        language_model_name: str | None,
        model_provider_api_key: str | None = None,
        embeddings_model_name: str | None = None,
):
    experiment_name = create_experiment_name(
        base_name='domain_classification',
        dataset_name=dataset_name.value,
        domain_classification_prompt_type=domain_classification_prompt_type,
        language_model_name=language_model_name,
        embeddings_model_name=embeddings_model_name,
    )

    config = Config(
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
    )
    dataset = AccuracyDatasetFactory.create(
        data_root_dir=config.data_root_dir,
        dataset_name=dataset_name
    ).get_dataset()

    language_model = (
        LanguageModelFactory
        .create(config)
        .get_language_model()
    )
    embeddings_model = (
        EmbeddingsModelFactory
        .create(config)
        .get_embeddings_model()
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        **config.vector_store,
    )

    domain_classification_prompt = (
        PromptsFactory
        .get_domain_classification_prompt(
            domain_classification_prompt_type
        )
    )

    domain_pipeline_nodes = [
        SimpleDocumentsRetrievalNode(
            vector_store=vector_store,
        ),
        MostRelevantDocumentAugmentationNode(),
        FakeQuestionAnswerNode()
    ]
    workflow = (
        QAWorkflowBuilder.create()
        .with_domain_routing_node(
            LLMDomainRoutingNode(
                language_model=language_model,
                prompt=domain_classification_prompt
            )
        )
        .add_sport_domain_nodes(domain_pipeline_nodes)
        .add_medicine_domain_nodes(domain_pipeline_nodes)
        .add_other_domain_nodes(domain_pipeline_nodes)
        .build()
    )

    await evaluate_domain_classification(
        dataset=dataset,
        experiment_name=experiment_name,
        workflow=workflow,
    )
