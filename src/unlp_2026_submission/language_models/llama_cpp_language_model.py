from typing import Any, TypedDict, Unpack, Optional, List, Dict, Union, Tuple
from pathlib import Path
import os
import json
import fnmatch
import re

from langchain_community.chat_models import ChatLlamaCpp
from huggingface_hub import hf_hub_download, HfFileSystem
from huggingface_hub.utils import validate_repo_id, enable_progress_bars

from unlp_2026_submission.config import Config


class LlamaCppLanguageModelKArgs(TypedDict, total=False):
    n_ctx: int
    """Token context window."""
    n_parts: int
    """Number of parts to split the model into.
    If -1, the number of parts is automatically determined."""
    seed: int
    """Seed. If -1, a random seed is used."""
    f16_kv: bool
    """Use half-precision for key/value cache."""
    logits_all: bool
    """Return logits for all tokens, not just the last token."""
    vocab_only: bool
    """Only load the vocabulary, no weights."""
    use_mlock: bool
    """Force system to keep model in RAM."""
    n_threads: Optional[int]
    """Number of threads to use.
    If None, the number of threads is automatically determined."""
    n_batch: int
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""
    n_gpu_layers: Optional[int]
    """Number of layers to be loaded into gpu memory. Default None."""
    suffix: Optional[str]
    """A suffix to append to the generated text. If None, no suffix is appended."""
    max_tokens: int
    """The maximum number of tokens to generate."""
    temperature: float
    """The temperature to use for sampling."""
    top_p: float
    """The top-p value to use for sampling."""
    logprobs: Optional[int]
    """The number of logprobs to return. If None, no logprobs are returned."""
    echo: bool
    """Whether to echo the prompt."""
    stop: Optional[List[str]]
    """A list of strings to stop generation when encountered."""
    repeat_penalty: float
    """The penalty to apply to repeated tokens."""
    top_k: int
    """The top-k value to use for sampling."""
    last_n_tokens_size: int
    """The number of tokens to look back when applying the repeat_penalty."""
    use_mmap: bool
    """Whether to keep the model loaded in RAM"""
    rope_freq_scale: float
    """Scale factor for rope sampling."""
    rope_freq_base: float
    """Base frequency for rope sampling."""
    model_kwargs: Dict[str, Any]
    """Any additional parameters to pass to llama_cpp.Llama."""
    streaming: bool
    """Whether to stream the results, token by token."""
    grammar_path: Optional[Union[str, Path]]
    """
    grammar_path: Path to the .gbnf file that defines formal grammars
    for constraining model outputs. For instance, the grammar can be used
    to force the model to generate valid JSON or to speak exclusively in emojis. At most
    one of grammar_path and grammar should be passed in.
    """
    grammar: Any
    """
    grammar: formal grammar for constraining model outputs. For instance, the grammar 
    can be used to force the model to generate valid JSON or to speak exclusively in 
    emojis. At most one of grammar_path and grammar should be passed in.
    """
    verbose: bool
    """Print verbose output to stderr."""

def parse_hf_repo_and_filename(ref: str) -> Tuple[str, Optional[str]]:
    """
    Returns (repo_id, filename)

    Examples:
        - org/repo
        - org/repo/file.gguf
        - https://huggingface.co/org/repo/blob/main/file.gguf
    """
    ref = ref.strip()

    # remove HF URL if present
    ref = re.sub(r"^https?://huggingface\.co/", "", ref)
    ref = ref.replace("/blob/main/", "/").replace("/resolve/main/", "/")

    parts = [p for p in ref.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Invalid Hugging Face reference: {ref}")

    repo_id = f"{parts[0]}/{parts[1]}"
    filename = parts[2] if len(parts) > 2 else None

    return repo_id, filename

def _resolve_model_hf_hub(
        config: Config,
        repo_id: str,
        filename: Optional[str],
        additional_files: Optional[List] = None,
):
    """
    --------------------------------------------------------
    THIS CODE COPIED FROM llama-cpp-python because LangChain hasn't implemented this method yet.
    ---------------------------------------------------------
    Create a Llama model from a pretrained model name or path.
    This method requires the huggingface-hub package.
    You can install it with `pip install huggingface-hub`.

    Args:
        repo_id: The model repo id.
        filename: A filename or glob pattern to match the model file in the repo.
        additional_files: A list of filenames or glob patterns to match additional model files in the repo.
        **kwargs: Additional keyword arguments to pass to the ChatLlamaCpp constructor.

    Returns:
        Model path"""

    validate_repo_id(repo_id)

    local_dir = config.downloaded_models_dir
    cache_dir = config.downloaded_models_cache_dir

    hffs = HfFileSystem()
    enable_progress_bars()

    files = [
        file["name"] if isinstance(file, dict) else file
        for file in hffs.ls(repo_id, recursive=True)
    ]

    # split each file into repo_id, subfolder, filename
    file_list: List[str] = []
    for file in files:
        rel_path = Path(file).relative_to(repo_id)
        file_list.append(str(rel_path))

    # find the only/first shard file:
    matching_files = [file for file in file_list if fnmatch.fnmatch(file, filename)]  # type: ignore

    if len(matching_files) == 0:
        raise ValueError(
            f"No file found in {repo_id} that match {filename}\n\n"
            f"Available Files:\n{json.dumps(file_list)}"
        )

    if len(matching_files) > 1:
        raise ValueError(
            f"Multiple files found in {repo_id} matching {filename}\n\n"
            f"Available Files:\n{json.dumps(files)}"
        )

    (matching_file,) = matching_files

    subfolder = str(Path(matching_file).parent)
    filename = Path(matching_file).name

    # download the file
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=local_dir,
        cache_dir=cache_dir,
    )

    if additional_files:
        for additonal_file_name in additional_files:
            # find the additional shard file:
            matching_additional_files = [file for file in file_list if fnmatch.fnmatch(file, additonal_file_name)]

            if len(matching_additional_files) == 0:
                raise ValueError(
                    f"No file found in {repo_id} that match {additonal_file_name}\n\n"
                    f"Available Files:\n{json.dumps(file_list)}"
                )

            if len(matching_additional_files) > 1:
                raise ValueError(
                    f"Multiple files found in {repo_id} matching {additonal_file_name}\n\n"
                    f"Available Files:\n{json.dumps(files)}"
                )

            (matching_additional_file,) = matching_additional_files

            # download the additional file
            hf_hub_download(
                repo_id=repo_id,
                filename=matching_additional_file,
                subfolder=subfolder,
                local_dir=local_dir,
                cache_dir=cache_dir,
            )

    if local_dir is None:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            local_dir=local_dir,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    else:
        model_path = os.path.join(local_dir, filename)

    return model_path

class LlamaCppLanguageModel(ChatLlamaCpp):
    @staticmethod
    def create(
            config: Config,
            **kwargs: Unpack[LlamaCppLanguageModelKArgs],
    ):
        if config.model_provider_api_key:
            if os.environ.get("HF_TOKEN") is None:
                os.environ["HF_TOKEN"] = config.model_provider_api_key

        repo_id, filename = parse_hf_repo_and_filename(config.language_model_name)

        model_path = _resolve_model_hf_hub(
            config=config,
            repo_id=repo_id,
            filename=filename,
        )
        # loading the first file of a sharded GGUF loads all remaining shard files in the subfolder
        return ChatLlamaCpp(
            model_path=model_path,
            max_tokens=config.language_model_max_tokens,
            n_batch=config.language_model_n_batch,
            n_ctx=config.language_model_context_window,
            n_gpu_layers=config.language_model_n_gpu_layers,
            repeat_penalty=config.language_model_repeat_penalty,
            rope_freq_base=0.0,
            rope_freq_scale=0.0,
            stop=config.language_model_stop_tokens,
            temperature=config.language_model_temperature,
            top_p=config.language_model_top_p,
            top_k=config.language_model_top_k,
            model_kwargs={ "penalize_nl": False },
            **kwargs,
        )

    @staticmethod
    def is_compatible_model(language_model_name: str) -> bool:
        if language_model_name and language_model_name.lower().endswith(".gguf"):
            return True
        return False

