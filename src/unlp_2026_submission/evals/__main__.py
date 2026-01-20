import asyncio

from unlp_2026_submission.evals.datasets.datasets import (
    get_dataset
)
from unlp_2026_submission.workflow import WorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import OllamaLanguageModel
from evaluate import evaluate_accuracy

async def main():
    dataset = get_dataset()
    config = Config()
    language_model = OllamaLanguageModel.create(config)

    workflow = (
        WorkflowBuilder
        .create(config)
        .with_language_model(language_model)
        .build()
    )

    await evaluate_accuracy(
        dataset=dataset,
        experiment_name="answers_accuracy_experiment",
        workflow=workflow
    )


if __name__ == "__main__":
    asyncio.run(main())
