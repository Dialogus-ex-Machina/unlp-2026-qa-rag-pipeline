import asyncio

import typer
from typing import Annotated

from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.evals.create_experiment_name import create_experiment_name
from unlp_2026_submission.evals.faithfulness import (
    evaluate_answers_faithfulness,
    FaithfulnessDatasetFactory,
    FaithfulnessDatasetName
)
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.workflow import WorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import LanguageModelFactory

app = typer.Typer()

@app.command('faithfulness', help='Evaluate faithfulness of answers.')
def evaluate_faithfulness_command(
        dataset_name: Annotated[
            FaithfulnessDatasetName,
            typer.Option("--dataset", "-ds")
        ] = FaithfulnessDatasetName.FULL,
        language_model_name: Annotated[
            str,
            typer.Option("--model", "-m")
        ] = None,
        model_provider_api_key: Annotated[str, typer.Option("--api-key", "-key")] = None,
):
    """
        Run evaluation for a given metric
    """
    asyncio.run(
        _evaluate(
            dataset_name=dataset_name,
            language_model_name=language_model_name,
            model_provider_api_key=model_provider_api_key,
        )
    )


async def _evaluate(
        dataset_name: FaithfulnessDatasetName,
        language_model_name: str | None,
        model_provider_api_key: str | None = None,
):
    experiment_name = create_experiment_name(
        base_name='faithfulness',
        dataset_name=dataset_name,
        language_model_name=language_model_name,
    )

    config = Config(
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key
    )
    dataset = FaithfulnessDatasetFactory.create(
        config=config,
        dataset_name=dataset_name
    ).get_dataset()

    language_model, llama_index_language_model = (
        LanguageModelFactory
        .create(config)
        .get_language_model()
    )
    embeddings_model = OpenAIEmbeddingsModel.create(config)

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

    await evaluate_answers_faithfulness(
        dataset=dataset,
        experiment_name=experiment_name,
        workflow=workflow
    )
