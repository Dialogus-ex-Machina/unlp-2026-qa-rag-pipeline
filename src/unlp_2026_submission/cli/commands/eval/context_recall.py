import asyncio

import typer
from typing import Annotated
import logging

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.evals.accuracy import (
    AccuracyMetricName,
    AccuracyDatasetFactory,
    AccuracyDatasetName,
)
from unlp_2026_submission.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.evals.context_recall import evaluate_context_recall
from unlp_2026_submission.evals.create_experiment_name import create_experiment_name
from unlp_2026_submission.workflow.nodes import (
    MostRelevantDocumentAugmentationNode,
    SimpleDocumentsRetrievalNode,
    SimpleQuestionAnswerNode,
    LLMDomainRoutingNode
)
from unlp_2026_submission.workflow.prompts.domain_classification_prompt_type import DomainClassificationPromptType
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import LanguageModelFactory, JudgeLanguageModelFactory
from unlp_2026_submission.workflow.prompts import QAPromptType, PromptsFactory

app = typer.Typer()


@app.command('context-recall')
def evaluate_context_recall_command(
        metric: Annotated[
            AccuracyMetricName,
            typer.Argument()
        ] = AccuracyMetricName.ANSWERS,
        dataset_name: Annotated[
            AccuracyDatasetName,
            typer.Option("--dataset", "-ds")
        ] = AccuracyDatasetName.FULL,
        qa_prompt_type: Annotated[
            QAPromptType,
            typer.Option("--qa-prompt")
        ] = QAPromptType.SIMPLE,
        domain_classification_prompt_type: Annotated[
            DomainClassificationPromptType,
            typer.Option("--classify-prompt")
        ] = DomainClassificationPromptType.SIMPLE_UA,
        language_model_name: Annotated[str, typer.Option("--model", "-m")] = None,
        model_provider_api_key: Annotated[str, typer.Option("--api-key", "-key")] = None,
        embeddings_model_name: Annotated[str, typer.Option("--embeddings-model", "-em")] = None,
        judge_language_model_name: Annotated[
            str,
            typer.Option("--judge-model")
        ] = None,
        judge_model_provider_api_key: Annotated[
            str,
            typer.Option("--judge-api-key")
        ] = None,
        logging_level: Annotated[int, typer.Option("--logs", "-l")] = logging.INFO,
):
    """
        Evaluate context recall for a given dataset
    """
    logging.basicConfig(level=logging_level)

    asyncio.run(
        _evaluate(
            metric=metric,
            dataset_name=dataset_name,
            qa_prompt_type=qa_prompt_type,
            domain_classification_prompt_type=domain_classification_prompt_type,
            language_model_name=language_model_name,
            model_provider_api_key=model_provider_api_key,
            embeddings_model_name=embeddings_model_name,
            judge_language_model_name=judge_language_model_name,
            judge_model_provider_api_key=judge_model_provider_api_key,
        )
    )


async def _evaluate(
        metric: AccuracyMetricName,
        dataset_name: AccuracyDatasetName,
        qa_prompt_type: QAPromptType,
        domain_classification_prompt_type: DomainClassificationPromptType,
        language_model_name: str | None,
        model_provider_api_key: str | None = None,
        embeddings_model_name: str | None = None,
        judge_language_model_name: str | None = None,
        judge_model_provider_api_key: str | None = None,
):
    experiment_name = create_experiment_name(
        base_name='context-recall',
        metric=metric.value,
        dataset_name=dataset_name.value,
        qa_prompt_type=qa_prompt_type.value,
        domain_classification_prompt_type=domain_classification_prompt_type.value,
        language_model_name=language_model_name,
        embeddings_model_name=embeddings_model_name,
        judge_language_model_name=judge_language_model_name,
    )

    config = Config(
        qa_prompt_type=qa_prompt_type,
        domain_classification_prompt_type=domain_classification_prompt_type,
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
        judge_language_model_name=judge_language_model_name,
        judge_language_model_provider_api_key=judge_model_provider_api_key,
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
    judge_language_model = (
        JudgeLanguageModelFactory
        .create(config)
        .get_language_model()
    )

    qa_prompt = (
        PromptsFactory
        .get_qa_prompt(config.qa_prompt_type)
    )
    domain_classification_prompt = (
        PromptsFactory
        .get_domain_classification_prompt(
            config.domain_classification_prompt_type
        )
    )

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

    await evaluate_context_recall(
        dataset=dataset,
        experiment_name=experiment_name,
        workflow=workflow,
        judge_language_model=judge_language_model,
    )
