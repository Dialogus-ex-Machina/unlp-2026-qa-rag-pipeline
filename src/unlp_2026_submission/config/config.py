import os
from dotenv import load_dotenv
from pathlib import Path

from unlp_2026_submission.workflow.prompts.qa_prompt_type import QAPromptType
from .vector_store_config import VectorStoreConfig

load_dotenv()

class Config:
    language_model_name: str
    language_model_context_window: int
    language_model_max_tokens: int
    language_model_temperature: float
    language_model_top_p: float
    language_model_top_k: int
    language_model_repeat_penalty: float
    language_model_stop_tokens: list[str]
    language_model_n_batch: int
    language_model_n_gpu_layers: int

    model_provider_api_key: str
    judge_language_model_name: str
    judge_language_model_provider_api_key: str
    embeddings_model_name: str
    downloaded_models_dir: str
    downloaded_models_cache_dir: str
    data_root_dir: str
    qa_prompt_type: QAPromptType

    vector_store: VectorStoreConfig

    def __init__(
            self,
            language_model_name: str | None = None,
            language_model_context_window: int | None = None,
            language_model_max_tokens: int | None = None,
            language_model_temperature: float | None = None,
            language_model_top_p: float | None = None,
            language_model_top_k: int | None = None,
            language_model_repeat_penalty: float | None = None,
            language_model_stop_tokens: list[str] | None = None,
            language_model_n_batch: int | None = None,
            language_model_n_gpu_layers: int | None = None,
            model_provider_api_key: str | None = None,
            embeddings_model_name: str | None = None,
            judge_language_model_name: str | None = None,
            judge_language_model_provider_api_key: str | None = None,
            qa_prompt_type: QAPromptType | None = None,
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
            int(os.getenv('LANGUAGE_MODEL_CONTEXT_WINDOW', '4096')),
        )
        self.model_provider_api_key = self._resolve_value_with_priority(
            model_provider_api_key,
            os.getenv('MODEL_PROVIDER_API_KEY')
        )
        self.language_model_max_tokens = self._resolve_value_with_priority(
            language_model_max_tokens,
            int(os.getenv('LANGUAGE_MODEL_MAX_TOKENS', '2048'))
        )
        self.language_model_temperature = self._resolve_value_with_priority(
            language_model_temperature,
            float(os.getenv('LANGUAGE_MODEL_TEMPERATURE', '0.1'))
        )
        self.language_model_top_p = self._resolve_value_with_priority(
            language_model_top_p,
            float(os.getenv('LANGUAGE_MODEL_TOP_P', '0.9'))
        )
        self.language_model_top_k = self._resolve_value_with_priority(
            language_model_top_k,
            int(os.getenv('LANGUAGE_MODEL_TOP_K', '40'))
        )
        self.language_model_repeat_penalty = self._resolve_value_with_priority(
            language_model_repeat_penalty,
            float(os.getenv('LANGUAGE_MODEL_REPEAT_PENALTY', '1.0'))
        )
        self.language_model_stop_tokens = self._resolve_value_with_priority(
            language_model_stop_tokens,
            os.getenv('LANGUAGE_MODEL_STOP_TOKENS', '<eos> <end_of_turn>').split()
        )
        self.language_model_n_batch = self._resolve_value_with_priority(
            language_model_n_batch,
            int(os.getenv('LANGUAGE_MODEL_N_BATCH', '512'))
        )
        self.language_model_n_gpu_layers = self._resolve_value_with_priority(
            language_model_n_gpu_layers,
            int(os.getenv('LANGUAGE_MODEL_N_GPU_LAYERS', '-1'))
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
        self.qa_prompt_type = self._resolve_value_with_priority(
            qa_prompt_type,
            QAPromptType.SIMPLE
        )

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

        vector_store_path = os.path.join(
            project_src_dir,
            os.getenv('VECTOR_STORE_PATH', 'vector_dbs/qdrant_db')
        )
        collection_name = os.getenv('VECTOR_STORE_COLLECTION_NAME', 'default')

        self.vector_store = {
            'path': vector_store_path,
            'collection_name': collection_name
        }

    def _resolve_value_with_priority(self, *values):
        for v in values:
            if v is not None:
                return v
        return None
