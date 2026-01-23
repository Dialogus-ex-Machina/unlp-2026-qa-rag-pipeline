import asyncio

import typer
from typing import Annotated

from unlp_2026_submission.evals import (
    EvaluationFactory,
    EvaluationMetricName,
    EvaluationDatasetFactory,
    EvaluationDatasetName,
)
from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.workflow import WorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import (
    OllamaLanguageModel,
    LlamaOllamaLanguageModel
)

app = typer.Typer()

@app.command('eval')
def evaluate_command(
        metric: Annotated[EvaluationMetricName, typer.Argument()] = EvaluationMetricName.ANSWER_ACCURACY,
        dataset_name: Annotated[
            EvaluationDatasetName,
            typer.Option("--dataset", "-ds")
        ] = EvaluationDatasetName.FULL
):
    """
        Run evaluation for a given metric
    """
    asyncio.run(
        _evaluate(
            metric=metric,
            dataset_name=dataset_name
        )
    )


async def _evaluate(
        metric: EvaluationMetricName,
        dataset_name: EvaluationDatasetName,
):
    # TODO add dynamic experiment_name creation logic or pass via options
    dataset = EvaluationDatasetFactory.create(
        dataset_name=dataset_name
    ).get_dataset()

    config = Config()
    language_model = OllamaLanguageModel.create(config)
    embeddings_model = OpenAIEmbeddingsModel.create(config)
    llama_language_model = LlamaOllamaLanguageModel.create(config)

    knowledge_base = KnowledgeBase.load(
        language_model=llama_language_model,
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

    eval_factory = EvaluationFactory.create(metric)

    await eval_factory.run(
        dataset=dataset,
        experiment_name=f"{metric.value}_experiment",
        workflow=workflow
    )
