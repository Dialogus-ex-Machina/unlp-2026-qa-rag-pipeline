import os
from typing import Any
from typing_extensions import TypedDict, Unpack
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_index.llms.google_genai import GoogleGenAI

from unlp_2026_submission.config import Config

class GeminiLanguageModelKArgs(TypedDict, total=False):
    temperature: float
    max_tokens: int
    timeout: float | tuple[float, float] | Any
    max_retries: int

class GeminiLanguageModel(ChatGoogleGenerativeAI):
    @staticmethod
    def create(
            config: Config,
            **kwargs: Unpack[GeminiLanguageModelKArgs],
    ) -> ChatGoogleGenerativeAI:
        if not config.model_provider_api_key:
            raise ValueError("Model provider API key not found")

        if os.environ.get("GOOGLE_API_KEY") is None:
            os.environ["GOOGLE_API_KEY"] = config.model_provider_api_key

        return ChatGoogleGenerativeAI(
            model=config.language_model_name,
            **kwargs
        )

    @staticmethod
    def is_compatible_model(language_model_name: str) -> bool:
        if language_model_name and language_model_name.lower().startswith("gemini"):
            return True
        return False

class LlamaIndexGeminiLanguageModel(GoogleGenAI):
    @staticmethod
    def create(
            config: Config,
            **kwargs: Unpack[GeminiLanguageModelKArgs],
    ) -> GoogleGenAI:
        if not config.model_provider_api_key:
            raise ValueError("Model provider API key not found")

        if os.environ.get("GOOGLE_API_KEY") is None:
            os.environ["GOOGLE_API_KEY"] = config.model_provider_api_key

        return GoogleGenAI(
            model=config.language_model_name,
            **kwargs
        )