from langchain_ollama import ChatOllama

from unlp_2026_submission.config import Config


class OllamaLanguageModel(ChatOllama):
    @staticmethod
    def create(config: Config):
        return ChatOllama(
            model=config.language_model_name
        )
