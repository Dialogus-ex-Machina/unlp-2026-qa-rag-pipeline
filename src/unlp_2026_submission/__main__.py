import asyncio

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow.nodes import (
    MostRelevantDocumentAugmentationNode,
    SimpleDocumentsRetrievalNode,
    QuestionAnswerNode,
)
from unlp_2026_submission.workflow.prompts import PromptsFactory
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder


async def main():
    config = Config()

    language_model = (
        LanguageModelFactory
            .create(config)
            .get_language_model()
    )
    embeddings_model = OpenAIEmbeddingsModel.create(config)

    qa_prompt = (
        PromptsFactory
        .create(config.qa_prompt_type)
        .get_qa_prompt()
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        **config.vector_store,
    )

    workflow = (
        QAWorkflowBuilder.create()
        .with_documents_retrieval_node(
            SimpleDocumentsRetrievalNode(
                vector_store=vector_store,
            ),
        )
        .with_augmentation_node(
            MostRelevantDocumentAugmentationNode()
        )
        .with_question_answering_node(
            QuestionAnswerNode(
                language_model=language_model,
                prompt=qa_prompt,
            )
        )
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
