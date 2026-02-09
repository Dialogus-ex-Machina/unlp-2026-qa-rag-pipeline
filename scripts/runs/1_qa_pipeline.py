import random

from langchain_qdrant import QdrantVectorStore

from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import SentenceTransformerEmbeddingModel
from unlp_2026_submission.evals.accuracy import AccuracyDatasetFactory, AccuracyDatasetName
from unlp_2026_submission.language_models import LlamaCppLanguageModel
from unlp_2026_submission.workflow.nodes import (
    MostRelevantDocumentAugmentationNode,
    SimpleDocumentsRetrievalNode,
    QuestionAnswerNode,
)
from unlp_2026_submission.workflow.prompts.qa_prompt import QAPrompt
from unlp_2026_submission.workflow.qa_workflow_builder import QAWorkflowBuilder


def main():
    config = Config(
        language_model_name="Qwen/Qwen2-0.5B-Instruct-GGUF/qwen2-0_5b-instruct-q8_0.gguf",
        embeddings_model_name="bflhc/Octen-Embedding-0.6B",
    )

    language_model = LlamaCppLanguageModel.create(
        config=config,
    )
    embeddings_model = SentenceTransformerEmbeddingModel.create(config)

    qa_prompt = QAPrompt()

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

    dataset = AccuracyDatasetFactory.create(
        data_root_dir=config.data_root_dir,
        dataset_name=AccuracyDatasetName.FULL
    ).get_dataset()
    question = random.Random().choice(dataset)

    response = workflow.invoke(
        input={ 'question': question }
    )

    print('Response:', response)

if __name__ == "__main__":
    main()
