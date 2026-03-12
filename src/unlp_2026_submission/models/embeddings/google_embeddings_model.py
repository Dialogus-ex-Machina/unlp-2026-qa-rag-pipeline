import os
from typing import Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class GoogleEmbeddingsModel(GoogleGenerativeAIEmbeddings):
    @staticmethod
    def create(
            model_name: str,
            api_key: Optional[str],
    ):
        if not api_key and os.environ.get("GOOGLE_API_KEY") is None:
            raise ValueError("Model provider API key not found")

        if os.environ.get("GOOGLE_API_KEY") is None:
            os.environ["GOOGLE_API_KEY"] = api_key

        return GoogleEmbeddingsModel(
            model=model_name,
        )

    @staticmethod
    def is_compatible_model(embeddings_model_name: str) -> bool:
        if (
                embeddings_model_name and
                embeddings_model_name.lower().startswith("gemini-embedding")
        ):
            return True
        return False
