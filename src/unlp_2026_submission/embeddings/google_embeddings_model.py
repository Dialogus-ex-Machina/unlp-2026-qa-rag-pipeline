import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from unlp_2026_submission.config import Config

class GoogleEmbeddingsModel(GoogleGenerativeAIEmbeddings):
    @staticmethod
    def create(config: Config):
        if not config.model_provider_api_key:
            raise ValueError("Model provider API key not found")

        if os.environ.get("GOOGLE_API_KEY") is None:
            os.environ["GOOGLE_API_KEY"] = config.model_provider_api_key

        return GoogleEmbeddingsModel(
            model=config.embeddings_model_name,
        )
