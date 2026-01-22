import typer

app = typer.Typer()

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.language_models import LlamaOllamaLanguageModel
from unlp_2026_submission.knowledge_base import KnowledgeBaseBuilder


@app.command('create')
def create_knowledge_base_command():
    """
        Creates knowledge base.
        Uploads embeddings to vector store and builds index for retrieval of documents.
    """
    config = Config()

    embeddings_model = OpenAIEmbeddingsModel.create(config)
    llama_language_model = LlamaOllamaLanguageModel.create(config)

    KnowledgeBaseBuilder.build(
        language_model=llama_language_model,
        embeddings_model=embeddings_model,
        config=config.knowledge_base,
        should_persist=True,
    )