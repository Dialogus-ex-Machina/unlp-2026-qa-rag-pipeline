import asyncio

import typer
from typing import Annotated
import logging

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.evals.create_experiment_name import create_experiment_name
from unlp_2026_submission.evals.faithfulness import (
    evaluate_answers_faithfulness,
    FaithfulnessDatasetFactory,
    FaithfulnessDatasetName
)
from unlp_2026_submission.workflow.nodes import (
    MostRelevantDocumentAugmentationNode,
    SimpleDocumentsRetrievalNode,
    QuestionAnswerNode,
)
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder
from unlp_2026_submission.config import Config
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow.prompts import QAPromptType, PromptsFactory

app = typer.Typer()


@app.command('faithfulness', help='Evaluate faithfulness of answers.')
def evaluate_faithfulness_command(
        dataset_name: Annotated[
            FaithfulnessDatasetName,
            typer.Option("--dataset", "-ds")
        ] = FaithfulnessDatasetName.FULL,
        qa_prompt_type: Annotated[
            QAPromptType,
            typer.Option("--qa-prompt")
        ] = QAPromptType.SIMPLE,
        language_model_name: Annotated[
            str,
            typer.Option("--model", "-m")
        ] = None,
        model_provider_api_key: Annotated[str, typer.Option("--api-key", "-key")] = None,
        embeddings_model_name: Annotated[str, typer.Option("--embeddings-model", "-em")] = None,
        logging_level: Annotated[int, typer.Option("--logs", "-l")] = logging.INFO,
):
    """
        Evaluate faithfulness for a given dataset
    """
    logging.basicConfig(level=logging_level)

    asyncio.run(
        _evaluate(
            dataset_name=dataset_name,
            qa_prompt_type=qa_prompt_type,
            language_model_name=language_model_name,
            model_provider_api_key=model_provider_api_key,
            embeddings_model_name=embeddings_model_name,
        )
    )


async def _evaluate(
        dataset_name: FaithfulnessDatasetName,
        qa_prompt_type: QAPromptType,
        language_model_name: str | None,
        model_provider_api_key: str | None = None,
        embeddings_model_name: str | None = None,
):
    experiment_name = create_experiment_name(
        base_name='faithfulness',
        dataset_name=dataset_name.value,
        qa_prompt_type=qa_prompt_type.value,
        language_model_name=language_model_name,
        embeddings_model_name=embeddings_model_name,
    )

    config = Config(
        qa_prompt_type=qa_prompt_type,
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
    )
    dataset = FaithfulnessDatasetFactory.create(
        data_root_dir=config.data_root_dir,
        dataset_name=dataset_name
    ).get_dataset()

    language_model = (
        LanguageModelFactory
        .create(config)
        .get_language_model()
    )
    embeddings_model = (
        EmbeddingsModelFactory
        .create(config)
        .get_embeddings_model()
    )

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

    await evaluate_answers_faithfulness(
        dataset=dataset,
        experiment_name=experiment_name,
        workflow=workflow,
    )
