from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.config import Config
from unlp_2026_submission.models.embeddings import EmbeddingsModelFactory
from unlp_2026_submission.models.language_models import LanguageModelFactory
from unlp_2026_submission.rag.qa.nodes import (
    MostRelevantDocsContextCreationNode,
    SimpleRetrievalNode,
    SimpleQuestionAnswerNode,
    LLMDomainRoutingNode
)
from unlp_2026_submission.rag.qa.qa_workflow_builder import QAWorkflowBuilder
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName
from unlp_2026_submission.rag.qa.prompts import (
    QAPromptType,
    PromptsFactory,
    DomainClassificationPromptType,
)


@dataclass(frozen=True)
class InvokeResult:
    question: str
    response: object


def build_config(
    language_model_name: Optional[str],
    model_provider_api_key: Optional[str],
    embeddings_model_name: Optional[str] = None,
) -> Config:
    return Config(
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
    )


def build_workflow(
        config: Config,
        qa_prompt_type: QAPromptType,
        domain_classification_prompt_type: DomainClassificationPromptType,
):
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
        .get_qa_prompt(qa_prompt_type)
    )
    domain_classification_prompt = (
        PromptsFactory
        .get_domain_classification_prompt(
            domain_classification_prompt_type
        )
    )

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        **config.vector_store,
    )

    nodes = [
        LLMDomainRoutingNode(
            language_model=language_model,
            prompt=domain_classification_prompt
        ),
        SimpleRetrievalNode(
            vector_store=vector_store,
        ),
        MostRelevantDocsContextCreationNode(),
        SimpleQuestionAnswerNode(
            language_model=language_model,
            prompt=qa_prompt,
        )
    ]
    workflow = (
        QAWorkflowBuilder.create()
        .add_nodes(nodes)
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
    domain_classification_prompt_type: DomainClassificationPromptType,
    language_model_name: Optional[str],
    model_provider_api_key: Optional[str],
    embeddings_model_name: Optional[str] = None,
    seed: Optional[int] = None,
    question: Optional[str] = None,
    logging_level: Optional[int] = logging.INFO,
) -> InvokeResult:
    logging.basicConfig(level=logging_level)

    config = build_config(
        language_model_name=language_model_name,
        model_provider_api_key=model_provider_api_key,
        embeddings_model_name=embeddings_model_name,
    )
    workflow = build_workflow(
        config=config,
        qa_prompt_type=qa_prompt_type,
        domain_classification_prompt_type=domain_classification_prompt_type,
    )

    q = question or sample_question(
        config=config,
        dataset_name=dataset_name,
        seed=seed
    )
    response = workflow.invoke(input={"question": q})
    return InvokeResult(question=q, response=response)