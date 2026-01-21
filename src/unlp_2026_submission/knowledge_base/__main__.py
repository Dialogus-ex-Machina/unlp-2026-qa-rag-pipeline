from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
# from unlp_2026_submission.language_models import MamayLanguageModel
from unlp_2026_submission.language_models import LlamaOllamaLanguageModel
from unlp_2026_submission.knowledge_base import KnowledgeBase, KnowledgeBaseBuilder

config = Config()


embeddings_model = OpenAIEmbeddingsModel.create(config)
llama_language_model = LlamaOllamaLanguageModel.create(config)

knowledge_base = KnowledgeBaseBuilder.build(
    language_model=llama_language_model,
    embeddings_model=embeddings_model,
    config=config.knowledge_base,
    should_persist=True,
)

response = knowledge_base.retrieve_page("""
    Як рекомендовано приймати ретаболіл дорослим?
""")

print('response', len(response))
