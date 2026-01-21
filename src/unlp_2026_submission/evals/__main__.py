import asyncio

from unlp_2026_submission.embeddings import OpenAIEmbeddingsModel
from unlp_2026_submission.evals.datasets.datasets import (
    get_full_dataset,
    get_medical_domain_dataset
)
from unlp_2026_submission.evals.evaluate_documents_source_accuracy import evaluate_documents_source_accuracy
from unlp_2026_submission.evals.evaluate_documents_source_page_accuracy import evaluate_documents_source_page_accuracy
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.workflow import WorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import OllamaLanguageModel, LlamaOllamaLanguageModel
from unlp_2026_submission.evals.evaluate_answers_accuracy import evaluate_answers_accuracy
from unlp_2026_submission.evals.evaluate_composite_accuracy import evaluate_composite_accuracy

async def main():
    dataset = get_medical_domain_dataset()

    config = Config()
    language_model = OllamaLanguageModel.create(config)
    embeddings_model = OpenAIEmbeddingsModel.create(config)
    llama_language_model = LlamaOllamaLanguageModel.create(config)

    knowledge_base = KnowledgeBase.load(
        language_model=llama_language_model,
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

    # await evaluate_documents_source_page_accuracy(
    #     dataset=dataset,
    #     experiment_name="documents_source_page_accuracy",
    #     workflow=workflow
    # )

    # await evaluate_documents_source_accuracy(
    #     dataset=dataset,
    #     experiment_name="documents_source_accuracy",
    #     workflow=workflow
    # )

    # await evaluate_answers_accuracy(
    #     dataset=dataset,
    #     experiment_name="answers_accuracy",
    #     workflow=workflow
    # )

    await evaluate_composite_accuracy(
        dataset=dataset,
        experiment_name="composite_accuracy",
        workflow=workflow
    )


if __name__ == "__main__":
    asyncio.run(main())
