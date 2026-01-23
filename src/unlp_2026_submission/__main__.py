import asyncio

from unlp_2026_submission.config.config import Config
from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow import WorkflowBuilder


async def main():
    config = Config()

    language_model, llama_index_language_model = (
        LanguageModelFactory
            .create(config)
            .get_language_model()
    )
    embeddings_model = OpenAIEmbeddingsModel.create(config)

    knowledge_base = KnowledgeBase.load(
        llama_index_language_model=llama_index_language_model,
        embeddings_model=embeddings_model,
        config=config.knowledge_base,
        should_persist=True,
    )

    workflow = (
        WorkflowBuilder
        .create(config)
        .with_language_model(language_model)
        .with_knowledge_base(knowledge_base)
        .build()
    )

    response = await workflow.ainvoke(
        input={
            'question': {
                'question_id': 0,
                'question_text': 'Як рекомендовано приймати ретаболіл дорослим?',
                'answers': [
                    'внутрішньо',
                    'підшкірно',
                    'орально',
                    'внутрішньовенно',
                    'внутрішньом’язово',
                    'інгаляційно'
                ],
                'domain': 'domain_2',
                'n_pages': 5,
                'correct_answer': 'E',
                'doc_id': '4e779acee13fa6e0763fb33d1c83030b8e6ea33d.pdf',
                'page_num': 1,
            }
        }
    )

    print('Response:', response)

if __name__ == "__main__":
    asyncio.run(main())
