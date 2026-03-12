import os
from typing import Optional
from langchain_openai import OpenAIEmbeddings

class OpenAIEmbeddingsModel(OpenAIEmbeddings):
    @staticmethod
    def create(
            model_name: str,
            api_key: Optional[str],
    ):
        if not api_key and os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError("Model provider API key not found")

        if os.environ.get("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = api_key

        return OpenAIEmbeddingsModel(
            model=model_name
        )

    @staticmethod
    def is_compatible_model(embeddings_model_name: str) -> bool:
        if (
                embeddings_model_name and
                embeddings_model_name.lower().startswith("text-embedding")
        ):
            return True
        return False


