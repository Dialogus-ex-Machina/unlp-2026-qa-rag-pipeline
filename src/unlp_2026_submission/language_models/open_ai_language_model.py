import os
from typing import Any
from typing_extensions import TypedDict, Unpack
from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI

from unlp_2026_submission.config import Config

class OpenAILanguageModelKArgs(TypedDict, total=False):
    temperature: float
    max_tokens: int
    timeout: float | tuple[float, float] | Any
    max_retries: int

class OpenAILanguageModel(ChatOpenAI):
    @staticmethod
    def create(
            config: Config,
            **kwargs: Unpack[OpenAILanguageModelKArgs],
    ) -> ChatOpenAI:
        if not config.model_provider_api_key:
            raise ValueError("Model provider API key not found")

        if os.environ.get("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = config.model_provider_api_key

        return ChatOpenAI(
            model=config.language_model_name,
            **kwargs
        )

    @staticmethod
    def is_compatible_model(language_model_name: str) -> bool:
        if language_model_name and language_model_name.lower().startswith("gpt"):
            return True
        return False


class LlamaIndexOpenAILanguageModel(OpenAI):
    @staticmethod
    def create(
            config: Config,
            **kwargs: Unpack[OpenAILanguageModelKArgs],
    ) -> OpenAI:
        if not config.model_provider_api_key:
            raise ValueError("Model provider API key not found")

        if os.environ.get("OPENAI_API_KEY") is None:
            os.environ["OPENAI_API_KEY"] = config.model_provider_api_key

        return OpenAI(
            model=config.language_model_name,
            **kwargs
        )