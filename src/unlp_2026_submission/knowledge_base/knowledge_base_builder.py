from unlp_2026_submission.config import Config
from unlp_2026_submission.embeddings import EmbeddingsModel
from unlp_2026_submission.knowledge_base import KnowledgeBase
from unlp_2026_submission.knowledge_base.documents import SimpleDirectoryReader
from unlp_2026_submission.knowledge_base.ingestion_pipelines import IngestionPipeline
from unlp_2026_submission.language_models import LlamaIndexLanguageModel

class KnowledgeBaseBuilder:
    @staticmethod
    def build(
            llama_index_language_model: LlamaIndexLanguageModel,
            embeddings_model: EmbeddingsModel,
            config: Config,
            should_persist: bool = True,
    ):
        knowledge_base = KnowledgeBase.create_empty(
            llama_index_language_model=llama_index_language_model,
            embeddings_model=embeddings_model,
            config=config.knowledge_base,
            should_persist=should_persist,
        )

        reader = SimpleDirectoryReader(
            input_dir=config.data_root_dir,
            recursive=True,
            required_exts=['.pdf']
        )
        # Custom metadata extraction?
        documents = reader.load_data(
            # num_workers=4
        )

        ingestion_pipeline = IngestionPipeline.create(
            config=config.knowledge_base,
            vector_store=knowledge_base.vector_store,
            doc_store=knowledge_base.doc_store,
            embeddings_model=embeddings_model,
        )

        nodes = ingestion_pipeline.run(
            documents=documents,
            show_progress=True
        )

        print(f"Knowledge base build succeed. Added {len(nodes)} nodes.")

        return knowledge_base
