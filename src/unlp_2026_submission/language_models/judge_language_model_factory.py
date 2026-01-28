from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import (
    OpenAILanguageModel,
    GeminiLanguageModel,
    OpenAIJudgeLanguageModel,
    JudgeLanguageModel, GeminiJudgeLanguageModel,
)

class JudgeLanguageModelFactory:
    _config: Config

    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def create(config: Config):
        return JudgeLanguageModelFactory(config)

    def get_language_model(self) -> JudgeLanguageModel:
        if OpenAILanguageModel.is_compatible_model(
                self._config.judge_language_model_name
        ):
            return OpenAIJudgeLanguageModel.create(self._config)

        if GeminiLanguageModel.is_compatible_model(
                self._config.judge_language_model_name
        ):
            return GeminiJudgeLanguageModel.create(self._config)

        raise ValueError(f"Unsupported judge language model: {self._config.judge_language_model_name}")
