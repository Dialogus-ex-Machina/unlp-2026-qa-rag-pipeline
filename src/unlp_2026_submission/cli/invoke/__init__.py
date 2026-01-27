import typer
import rich
import random
from typing import Annotated

from unlp_2026_submission.config.config import Config
from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow import WorkflowBuilder
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName

app = typer.Typer()

@app.command('invoke')
def invoke_command(
        dataset_name: Annotated[
            AccuracyDatasetName,
            typer.Option("--dataset", "-ds")
        ] = AccuracyDatasetName.FULL,
        language_model_name: Annotated[str, typer.Option("--model", "-m")] = None,
        model_provider_api_key: Annotated[str, typer.Option("--api-key", "-key")] = None,
):
    """
        Run workflow for random sampled question from dataset.
    """

    config = Config(
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
    )
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

    dataset = AccuracyDatasetFactory.create(dataset_name).get_dataset()
    question = random.choice(dataset)

    response = workflow.invoke(
        input={ 'question': question }
    )

    rich.print('Result:', response)