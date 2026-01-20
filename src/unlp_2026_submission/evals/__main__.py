import asyncio

from unlp_2026_submission.evals.datasets.datasets import (
    get_full_dataset
)
from unlp_2026_submission.evals.evaluate_documents_source_accuracy import evaluate_documents_source_accuracy
from unlp_2026_submission.workflow import WorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import OllamaLanguageModel
from unlp_2026_submission.evals.evaluate_answers_accuracy import evaluate_answers_accuracy

async def main():
    dataset = get_full_dataset()
    config = Config()
    language_model = OllamaLanguageModel.create(config)

    workflow = (
        WorkflowBuilder
        .create(config)
        .with_language_model(language_model)
        .build()
    )

    await evaluate_documents_source_accuracy(
        dataset=dataset,
        experiment_name="documents_source_accuracy",
        workflow=workflow
    )

    # await evaluate_answers_accuracy(
    #     dataset=dataset,
    #     experiment_name="answers_accuracy",
    #     workflow=workflow
    # )


if __name__ == "__main__":
    asyncio.run(main())
