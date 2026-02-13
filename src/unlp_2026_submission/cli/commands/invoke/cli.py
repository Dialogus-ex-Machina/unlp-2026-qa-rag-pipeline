import logging
import typer
from typing import Annotated, Optional
from rich import print as rprint

from unlp_2026_submission.evals.accuracy import AccuracyDatasetName
from unlp_2026_submission.workflow.prompts import QAPromptType, DomainClassificationPromptType
from .service import run_invoke

app = typer.Typer(no_args_is_help=True)


@app.command("run")
def run_command(
    dataset_name: Annotated[
        AccuracyDatasetName,
        typer.Option("--dataset", "-d", help="Which dataset to sample from."),
    ] = AccuracyDatasetName.FULL,
    qa_prompt_type: Annotated[
        Optional[QAPromptType],
        typer.Option("--qa-prompt")
    ] = QAPromptType.ENG,
    domain_classification_prompt_type: Annotated[
        DomainClassificationPromptType,
        typer.Option("--classify-prompt")
    ] = DomainClassificationPromptType.ENG,
    language_model_name: Annotated[
        Optional[str],
        typer.Option("--model", "-m", help="Language model name."),
    ] = None,
    model_provider_api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", help="Provider API key."),
    ] = None,
    embeddings_model_name: Annotated[
        Optional[str],
        typer.Option("--embeddings-model", "-em")
    ] = None,
    seed: Annotated[
        Optional[int],
        typer.Option("--seed", help="Random seed for deterministic sampling."),
    ] = None,
    question: Annotated[
        Optional[str],
        typer.Option("--question", "-q", help="Run a specific question instead of sampling."),
    ] = None,
    logging_level: Annotated[
        Optional[int],
        typer.Option("--logs", "-l")
    ] = logging.INFO,
):
    """
    Run workflow for a sampled (or provided) question.
    """
    result = run_invoke(
        dataset_name=dataset_name,
        qa_prompt_type=qa_prompt_type,
        domain_classification_prompt_type=domain_classification_prompt_type,
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
        seed=seed,
        question=question,
        logging_level=logging_level,
    )

    rprint("[bold]Question:[/bold]", result.question)
    rprint("[bold]Result:[/bold]", result.response)