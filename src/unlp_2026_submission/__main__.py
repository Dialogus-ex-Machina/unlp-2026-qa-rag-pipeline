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
        input={
            'question': {
                "question_id": "2966ef8f-c873-45cc-ba2c-5eec1407bc38",
                "question_text": "Розв'яжіть рівняння (frac{x + 3}{3x - 2} - frac{x + 1}{3x + 2} = 0).",
                "answers": [
                    '\\(- \\frac{5}{4}\\)',
                    '\\(\\  - \\frac{1}{3}\\)',
                    '\\(\\  - 0,8\\)',
                    '\\(0,8\\)'
                ],
            }
        }
    )

    print('Response:', response)

if __name__ == "__main__":
    asyncio.run(main())
