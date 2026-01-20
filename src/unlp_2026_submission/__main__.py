import asyncio
from unlp_2026_submission.config.config import Config
from unlp_2026_submission.language_models import OllamaLanguageModel
from unlp_2026_submission.workflow import WorkflowBuilder

async def main():
    config = Config()

    language_model = OllamaLanguageModel.create(config)

    workflow = (
        WorkflowBuilder
        .create(config)
        .with_language_model(language_model)
        .build()
    )

    response = await workflow.ainvoke(
        input={'query': 'What is the capital of France?'}
    )

    print('Response:', response)

if __name__ == "__main__":
    asyncio.run(main())
