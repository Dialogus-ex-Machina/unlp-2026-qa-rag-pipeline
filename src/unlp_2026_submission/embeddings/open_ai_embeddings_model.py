import os
from langchain_openai import OpenAIEmbeddings
from unlp_2026_submission.config import Config

class OpenAIEmbeddingsModel(OpenAIEmbeddings):
    @staticmethod
    def create(config: Config):
        if not config.model_provider_api_key:
            raise ValueError("Model provider API key not found")

        if os.environ.get("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = config.model_provider_api_key

        return OpenAIEmbeddingsModel(
            model=config.embeddings_model_name
        )
