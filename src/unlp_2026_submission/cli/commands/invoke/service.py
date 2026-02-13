from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow.nodes import (
    MostRelevantDocumentAugmentationNode,
    SimpleDocumentsRetrievalNode,
    SimpleQuestionAnswerNode
)
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName
from unlp_2026_submission.workflow.prompts import QAPromptType, PromptsFactory


@dataclass(frozen=True)
class InvokeResult:
    question: str
    response: object


def build_config(
    qa_prompt_type: QAPromptType,
    language_model_name: Optional[str],
    model_provider_api_key: Optional[str],
    embeddings_model_name: Optional[str] = None,
) -> Config:
    return Config(
        qa_prompt_type=qa_prompt_type,
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
    )


def build_workflow(config: Config):
    language_model = (
        LanguageModelFactory.create(config).get_language_model()
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
            SimpleQuestionAnswerNode(
                language_model=language_model,
                prompt=qa_prompt,
            )
        )
        .build()
    )
    return workflow


def sample_question(
    config: Config,
    dataset_name: AccuracyDatasetName,
    seed: Optional[int] = None,
) -> str:
    dataset = AccuracyDatasetFactory.create(
        data_root_dir=config.data_root_dir,
        dataset_name=dataset_name
    ).get_dataset()
    if not dataset:
        raise ValueError(f"Dataset {dataset_name} is empty.")

    rng = random.Random(seed)
    return rng.choice(dataset)


def run_invoke(
    dataset_name: AccuracyDatasetName,
    qa_prompt_type: QAPromptType,
    language_model_name: Optional[str],
    model_provider_api_key: Optional[str],
    embeddings_model_name: Optional[str] = None,
    seed: Optional[int] = None,
    question: Optional[str] = None,
    logging_level: Optional[int] = logging.INFO,
) -> InvokeResult:
    logging.basicConfig(level=logging_level)

    config = build_config(
        qa_prompt_type=qa_prompt_type,
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
    )
    workflow = build_workflow(config)

    q = question or sample_question(
        config=config,
        dataset_name=dataset_name,
        seed=seed
    )
    response = workflow.invoke(input={"question": q})
    return InvokeResult(question=q, response=response)