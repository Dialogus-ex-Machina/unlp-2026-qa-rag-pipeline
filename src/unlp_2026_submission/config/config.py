import os
from dotenv import load_dotenv
from pathlib import Path

from unlp_2026_submission.config.knowledge_base_config import KnowledgeBaseConfig

load_dotenv()

class Config:
    language_model_name: str
    model_provider_api_key: str
    embeddings_model_name: str

    knowledge_base: KnowledgeBaseConfig

    def __init__(self):
        self.language_model_name = os.getenv('LANGUAGE_MODEL_NAME')
        self.model_provider_api_key = os.getenv('MODEL_PROVIDER_API_KEY')
        self.embeddings_model_name = os.getenv('EMBEDDINGS_MODEL_NAME')

        # **/unlp-2026-submission/src/unlp_2026_submission
        package_root_dir = Path(__file__).resolve().parents[1]
        # **/unlp-2026-submission/src
        project_root_dir = package_root_dir.parent

        kb_store_root_dir = os.path.join(
            project_root_dir,
            os.getenv('KB_DATA_ROOT_DIR', 'kb_data')
        )
        data_root_dir = os.getenv('INPUT_DATA_DIR', project_root_dir / "data")
        vector_store_path = os.path.join(
            kb_store_root_dir,
            os.getenv('KB_VECTOR_STORE_PATH', 'db')
        )
        context_path = os.path.join(
            kb_store_root_dir,
            os.getenv('KB_CONTEXT_PATH', 'context')
        )
        collection_name = os.getenv('KB_COLLECTION_NAME', 'default_collection')

        self.knowledge_base = KnowledgeBaseConfig(
            kb_store_root_dir=kb_store_root_dir,
            data_root_dir=data_root_dir,
            vector_store_path=vector_store_path,
            context_path=context_path,
            collection_name=collection_name
        )
