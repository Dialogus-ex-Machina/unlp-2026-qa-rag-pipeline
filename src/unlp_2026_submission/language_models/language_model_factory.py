import re
from typing import Tuple, Optional

from unlp_2026_submission.config import Config
from .gemini_language_model import (
    GeminiLanguageModel,
    LlamaIndexGeminiLanguageModel
)
from .open_ai_language_model import (
    OpenAILanguageModel,
    LlamaIndexOpenAILanguageModel,
)
from .llama_cpp_language_model import (
    LlamaCppLanguageModel,
    LlamaIndexLlamaCppLanguageModel,
)
from .hugging_face_language_model import (
    HuggingFaceLanguageModel,
    LlamaIndexHuggingFaceLanguageModel,
)
from .language_model import LanguageModel, LlamaIndexLanguageModel


class LanguageModelFactory:
    _config: Config

    def __init__(self, config: Config):
        self._config = config

    @staticmethod
    def create(config: Config):
        return LanguageModelFactory(config)

    def get_language_model(self) -> tuple[LanguageModel, LlamaIndexLanguageModel]:
        if LlamaCppLanguageModel.is_compatible_model(self._config.language_model_name):
            repo_id, filename = self._parse_hf_repo_and_filename(self._config.language_model_name)

            language_model = LlamaCppLanguageModel.create_from_hf_hub(
                config=self._config,
                repo_id=repo_id,
                filename=filename
            )
            llama_index_language_model = LlamaIndexLlamaCppLanguageModel.create_from_hf_hub(
                config=self._config,
                repo_id=repo_id,
                filename=filename
            )

            return language_model, llama_index_language_model

        if OpenAILanguageModel.is_compatible_model(self._config.language_model_name):
            language_model = OpenAILanguageModel.create(self._config)
            llama_index_language_model = LlamaIndexOpenAILanguageModel.create(self._config)

            return language_model, llama_index_language_model

        if GeminiLanguageModel.is_compatible_model(self._config.language_model_name):
            language_model = GeminiLanguageModel.create(self._config)
            llama_index_language_model = LlamaIndexGeminiLanguageModel.create(self._config)

            return language_model, llama_index_language_model

        language_model = HuggingFaceLanguageModel.create(self._config)
        llama_index_language_model = LlamaIndexHuggingFaceLanguageModel.create(self._config)

        return language_model, llama_index_language_model

    def _parse_hf_repo_and_filename(self, ref: str) -> Tuple[str, Optional[str]]:
        """
        Returns (repo_id, filename)

        Examples:
          - org/repo
          - org/repo/file.gguf
          - https://huggingface.co/org/repo/blob/main/file.gguf
        """
        ref = ref.strip()

        # remove HF URL if present
        ref = re.sub(r"^https?://huggingface\.co/", "", ref)
        ref = ref.replace("/blob/main/", "/").replace("/resolve/main/", "/")

        parts = [p for p in ref.split("/") if p]
        if len(parts) < 2:
            raise ValueError(f"Invalid Hugging Face reference: {ref}")

        repo_id = f"{parts[0]}/{parts[1]}"
        filename = parts[2] if len(parts) > 2 else None

        return repo_id, filename
