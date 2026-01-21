from dataclasses import dataclass

@dataclass
class KnowledgeBaseConfig:
    kb_store_root_dir: str
    data_root_dir: str
    vector_store_path: str
    context_path: str
    collection_name: str
