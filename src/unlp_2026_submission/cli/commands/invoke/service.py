from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.language_models import LanguageModelFactory
from unlp_2026_submission.workflow import WorkflowBuilder
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName
from unlp_2026_submission.workflow.prompts import QAPromptType


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
    language_model, llama_index_language_model = (
        LanguageModelFactory.create(config).get_language_model()
    )
    embeddings_model = (
        EmbeddingsModelFactory
        .create(config)
        .get_embeddings_model()
    )

    knowledge_base = KnowledgeBase.load(
        llama_index_language_model=llama_index_language_model,
        embeddings_model=embeddings_model,
        config=config.knowledge_base,
    )

    workflow = (
        WorkflowBuilder.create(config)
        .with_language_model(language_model)
        .with_knowledge_base(knowledge_base)
        .build()
    )
    return workflow


def sample_question(
    config: Config,
    dataset_name: AccuracyDatasetName,
    seed: Optional[int] = None,
) -> str:
    dataset = AccuracyDatasetFactory.create(
        config=config,
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