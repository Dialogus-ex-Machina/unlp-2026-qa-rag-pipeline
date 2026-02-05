import asyncio

import typer
from typing import Annotated
import logging

from unlp_2026_submission.evals.accuracy import (
    AccuracyEvaluationFactory,
    AccuracyMetricName,
    AccuracyDatasetFactory,
    AccuracyDatasetName,
)
from unlp_2026_submission.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.evals.create_experiment_name import create_experiment_name
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.workflow.workflow_builder import WorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow.prompts import QAPromptType

app = typer.Typer()


@app.command('accuracy')
def evaluate_accuracy_command(
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
        language_model_name: Annotated[str, typer.Option("--model", "-m")] = None,
        model_provider_api_key: Annotated[str, typer.Option("--api-key", "-key")] = None,
        embeddings_model_name: Annotated[str, typer.Option("--embeddings-model", "-em")] = None,
        logging_level: Annotated[int, typer.Option("--logs", "-l")] = logging.INFO,
):
    """
        Evaluate accuracy for a given metric
    """
    logging.basicConfig(level=logging_level)

    asyncio.run(
        _evaluate(
            metric=metric,
            dataset_name=dataset_name,
            qa_prompt_type=qa_prompt_type,
            language_model_name=language_model_name,
            model_provider_api_key=model_provider_api_key,
            embeddings_model_name=embeddings_model_name,
        )
    )


async def _evaluate(
        metric: AccuracyMetricName,
        dataset_name: AccuracyDatasetName,
        qa_prompt_type: QAPromptType,
        language_model_name: str | None,
        model_provider_api_key: str | None = None,
        embeddings_model_name: str | None = None,
):
    experiment_name = create_experiment_name(
        base_name='accuracy',
        metric=metric.value,
        dataset_name=dataset_name.value,
        qa_prompt_type=qa_prompt_type.value,
        language_model_name=language_model_name,
        embeddings_model_name=embeddings_model_name,
    )

    config = Config(
        qa_prompt_type=qa_prompt_type,
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
    )
    dataset = AccuracyDatasetFactory.create(
        config=config,
        dataset_name=dataset_name
    ).get_dataset()

    language_model, llama_index_language_model = (
        LanguageModelFactory
        .create(config)
        .get_language_model()
    )
    embeddings_model = (
        EmbeddingsModelFactory
        .create(config)
        .get_embeddings_model()
    )

    knowledge_base = KnowledgeBase.load(
        llama_index_language_model=llama_index_language_model,
        embeddings_model=embeddings_model,
        config=config.knowledge_base,
    )

    workflow = (
        WorkflowBuilder
        .create(config)
        .with_language_model(language_model)
        .with_knowledge_base(knowledge_base)
        .build()
    )

    eval_factory = AccuracyEvaluationFactory.create(metric)

    await eval_factory.run(
        dataset=dataset,
        experiment_name=experiment_name,
        workflow=workflow,
    )
