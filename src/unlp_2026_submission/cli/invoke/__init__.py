import typer
import rich
import random
from typing import Annotated

from unlp_2026_submission.config.config import Config
from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.language_models import (
    OllamaLanguageModel,
    LlamaOllamaLanguageModel
)
from unlp_2026_submission.workflow import WorkflowBuilder
from unlp_2026_submission.evals import EvaluationDatasetFactory, EvaluationDatasetName

app = typer.Typer()

@app.command('invoke')
def invoke_command(
        dataset_name: Annotated[
            EvaluationDatasetName,
            typer.Option("--dataset", "-ds")
        ] = EvaluationDatasetName.FULL
):
    """
        Run workflow for random sampled question from dataset.
    """

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

    dataset = EvaluationDatasetFactory.create(dataset_name).get_dataset()
    question = random.choice(dataset)

    response = workflow.invoke(
        input={ 'question': question }
    )

    rich.print('Result:', response)