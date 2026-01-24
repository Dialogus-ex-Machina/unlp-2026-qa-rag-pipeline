import typer
from typing import Annotated

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.knowledge_base import KnowledgeBaseBuilder

app = typer.Typer()

@app.command('create')
def create_knowledge_base_command(
        language_model_name: Annotated[str, typer.Option("--model", "-m")] = None,
        model_provider_api_key: Annotated[str, typer.Option("--api-key", "-key")] = None,
):
    """
        Creates knowledge base.
        Uploads embeddings to vector store and builds index for retrieval of documents.
    """
    config = Config(
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
    )

    embeddings_model = OpenAIEmbeddingsModel.create(config)
    _, llama_index_language_model = (
        LanguageModelFactory
        .create(config)
        .get_language_model()
    )

    KnowledgeBaseBuilder.build(
        llama_index_language_model=llama_index_language_model,
        embeddings_model=embeddings_model,
        config=config.knowledge_base,
        should_persist=True,
    )