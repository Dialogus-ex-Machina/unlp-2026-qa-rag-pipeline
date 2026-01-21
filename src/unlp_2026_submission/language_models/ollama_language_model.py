from langchain_ollama import ChatOllama
from llama_index.llms.ollama import Ollama
from typing import Any, TypedDict, Unpack

from unlp_2026_submission.config import Config


class OllamaLanguageModelKArgs(TypedDict, total=False):
    temperature: float
    max_tokens: int
    timeout: float | tuple[float, float] | Any
    max_retries: int

class OllamaLanguageModel(ChatOllama):
    @staticmethod
    def create(
            config: Config,
            **kwargs: Unpack[OllamaLanguageModelKArgs],
    ):
        return ChatOllama(
            model=config.language_model_name,
            **kwargs
        )

class LlamaOllamaLanguageModel(Ollama):
    @staticmethod
    def create(
            config: Config,
            **kwargs: Unpack[OllamaLanguageModelKArgs],
    ) -> Ollama:
        return Ollama(
            model=config.language_model_name,
            **kwargs
        )
