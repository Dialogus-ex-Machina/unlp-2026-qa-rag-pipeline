import logging
import typer
from typing import Annotated, Optional

from unlp_2026_submission.config import Config
from unlp_2026_submission.models.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.models.language_models import LanguageModelFactory

app = typer.Typer(no_args_is_help=True)


@app.command('create')
def create_knowledge_base_command(
        language_model_name: Annotated[Optional[str], typer.Option("--model", "-m")] = None,
        model_provider_api_key: Annotated[Optional[str], typer.Option("--api-key", "-key")] = None,
        embeddings_model_name: Annotated[Optional[str], typer.Option("--embeddings-model", "-em")] = None,
        logging_level: Annotated[Optional[int], typer.Option("--logs", "-l")] = logging.INFO,
):
    """
        Creates knowledge base.
        Uploads embeddings to vector store and builds index for retrieval of documents.
    """
    logging.basicConfig(level=logging_level)

    config = Config(
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
    )
    embeddings_model = (
        EmbeddingsModelFactory
        .create(config)
        .get_embeddings_model()
    )
    # TODO add another implementation
