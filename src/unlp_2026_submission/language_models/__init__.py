from langchain_core.language_models import BaseChatModel as LanguageModel
from llama_index.core.llms import LLM as LlamaLanguageModel

from unlp_2026_submission.language_models.ollama_language_model import (
    OllamaLanguageModel,
    LlamaOllamaLanguageModel
)
from unlp_2026_submission.language_models.gemini_language_model import GeminiLanguageModel
from unlp_2026_submission.language_models.open_ai_language_model import OpenAILanguageModel
