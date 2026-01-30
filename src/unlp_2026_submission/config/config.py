import os
from dotenv import load_dotenv
from pathlib import Path

from unlp_2026_submission.workflow.prompts import QAPromptType
from .knowledge_base_config import KnowledgeBaseConfig

load_dotenv()

class Config:
    language_model_name: str
    language_model_context_window: int
    model_provider_api_key: str
    judge_language_model_name: str
    judge_language_model_provider_api_key: str
    embeddings_model_name: str
    downloaded_models_dir: str
    downloaded_models_cache_dir: str
    data_root_dir: str

    qa_prompt_type: QAPromptType

    knowledge_base: KnowledgeBaseConfig

    def __init__(
            self,
            language_model_name: str | None = None,
            language_model_context_window: int | None = None,
            model_provider_api_key: str | None = None,
            embeddings_model_name: str | None = None,
            judge_language_model_name: str | None = None,
            judge_language_model_provider_api_key: str | None = None,
            qa_prompt_type: QAPromptType = QAPromptType.SIMPLE,
    ):
        self.language_model_name = self._resolve_value_with_priority(
            language_model_name,
            os.getenv(
                'LANGUAGE_MODEL_NAME',
                'INSAIT-Institute/MamayLM-Gemma-3-4B-IT-v1.0-GGUF/MamayLM-Gemma-3-4B-IT-v1.0.Q8_0.gguf'
            ),
        )
        self.language_model_context_window = self._resolve_value_with_priority(
            language_model_context_window,
            int(os.getenv('LANGUAGE_MODEL_CONTEXT_WINDOW', '10000')),
        )
        self.model_provider_api_key = self._resolve_value_with_priority(
            model_provider_api_key,
            os.getenv('MODEL_PROVIDER_API_KEY')
        )
        self.embeddings_model_name = self._resolve_value_with_priority(
            embeddings_model_name,
            os.getenv('EMBEDDINGS_MODEL_NAME', 'bflhc/Octen-Embedding-0.6B'),
        )
        self.judge_language_model_name = self._resolve_value_with_priority(
            judge_language_model_name,
            os.getenv('JUDGE_LANGUAGE_MODEL_NAME', 'gemini-3-pro-preview')
        )
        self.judge_language_model_provider_api_key = self._resolve_value_with_priority(
            judge_language_model_provider_api_key,
            os.getenv('JUDGE_LANGUAGE_MODEL_PROVIDER_API_KEY')
        )
        self.qa_prompt_type = qa_prompt_type

        # **/unlp-2026-submission/src/unlp_2026_submission
        package_root_dir = Path(__file__).resolve().parents[1]
        # **/unlp-2026-submission/src
        project_src_dir = package_root_dir.parent
        project_root_dir = project_src_dir.parent

        self.data_root_dir = os.getenv('INPUT_DATA_DIR', project_src_dir / "data")

        hf_home_dir = os.getenv('HF_HOME', os.path.join(project_root_dir, "hf_cache"))
        if os.environ.get("HF_HOME") is None:
            os.environ["HF_HOME"] = hf_home_dir

        self.downloaded_models_dir = os.getenv(
            'DOWNLOADED_MODELS_DIR',
            os.path.join(hf_home_dir, "models")
        )
        self.downloaded_models_cache_dir = os.getenv(
            'DOWNLOADED_MODELS_CACHE_DIR',
            os.path.join(hf_home_dir, "models/.cache")
        )

        kb_store_root_dir = os.path.join(
            project_src_dir,
            os.getenv('KB_DATA_ROOT_DIR', 'kb_data')
        )
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
            vector_store_path=vector_store_path,
            context_path=context_path,
            collection_name=collection_name
        )

    def _resolve_value_with_priority(self, *values):
        for v in values:
            if v is not None:
                return v
        return None
