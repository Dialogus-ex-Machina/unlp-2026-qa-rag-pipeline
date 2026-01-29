from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class EmbeddingsModelSpec:
    provider: str
    api_key: str
    model_name: str

    cache_folder: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    encode_kwargs: dict[str, Any] = field(default_factory=dict)
    query_encode_kwargs: dict[str, Any] = field(default_factory=dict)

    multi_process: bool = False
    show_progress: bool = True
