import asyncio

import typer
from typing import Annotated
import logging

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.evals.accuracy import (
    AccuracyEvaluationFactory,
    AccuracyMetricName,
    AccuracyDatasetFactory,
    AccuracyDatasetName,
)
from unlp_2026_submission.models.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.evals.create_experiment_name import create_experiment_name
from unlp_2026_submission.rag.qa.nodes import (
    SimpleQuestionAnswerNode,
    SimpleRetrievalNode,
    TopKDocsContextCreationNode,
    LogprobRerankerNode,
)
from unlp_2026_submission.rag.qa.qa_workflow_builder import QAWorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.models.language_models import LanguageModelFactory
from unlp_2026_submission.rag.qa.prompts import (
    QAPromptType,
    PromptsFactory,
    DomainClassificationPromptType,
)

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
        ] = QAPromptType.UKR,
        domain_classification_prompt_type: Annotated[
            DomainClassificationPromptType,
            typer.Option("--classify-prompt")
        ] = DomainClassificationPromptType.UKR,
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
            domain_classification_prompt_type=domain_classification_prompt_type,
            language_model_name=language_model_name,
            model_provider_api_key=model_provider_api_key,
            embeddings_model_name=embeddings_model_name,
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
):
    experiment_name = create_experiment_name(
        base_name='accuracy',
        metric=metric.value,
        dataset_name=dataset_name.value,
        qa_prompt_type=qa_prompt_type.value,
        domain_classification_prompt_type=domain_classification_prompt_type.value,
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

    qa_prompt = (
        PromptsFactory
        .get_qa_prompt(qa_prompt_type)
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        **config.vector_store,
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
        )
    ]
    workflow = (
        QAWorkflowBuilder.create()
        .add_nodes(nodes)
        .build()
    )

    eval_factory = AccuracyEvaluationFactory.create(metric)

    await eval_factory.run(
        dataset=dataset,
        experiment_name=experiment_name,
        workflow=workflow,
    )
