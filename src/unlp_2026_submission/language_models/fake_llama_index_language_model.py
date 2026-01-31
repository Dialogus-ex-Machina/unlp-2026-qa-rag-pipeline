from typing import Any, Sequence

from llama_index.core.base.llms.types import (
    CompletionResponseAsyncGen,
    ChatMessage,
    ChatResponseAsyncGen,
    CompletionResponse,
    ChatResponse,
    CompletionResponseGen,
    ChatResponseGen,
    LLMMetadata
)

from .language_model import  LlamaIndexLanguageModel

class FakeLlamaIndexLanguageModel(LlamaIndexLanguageModel):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata()

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        pass

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        pass

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        pass

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        pass

    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        pass

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        pass

    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        pass

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        pass