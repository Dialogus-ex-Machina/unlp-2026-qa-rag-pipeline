from unlp_2026_submission.config import Config
from .llama_cpp_language_model import LlamaCppLanguageModel
from .hugging_face_language_model import HuggingFaceLanguageModel
from .language_model import LanguageModel


class LanguageModelFactory:
    _config: Config

    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def create(config: Config):
        return LanguageModelFactory(config)

    def get_language_model(self) -> LanguageModel:
        if LlamaCppLanguageModel.is_compatible_model(self._config.language_model_name):
            language_model = LlamaCppLanguageModel.create(self._config)

            return language_model

        language_model = HuggingFaceLanguageModel.create(self._config)

        return language_model
