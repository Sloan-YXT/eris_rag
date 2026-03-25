"""Configuration management with YAML loading and environment variable expansion."""

import os
import re
from pathlib import Path
from typing import Any

import yaml


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${ENV_VAR} patterns in string values."""
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


class Config:
    """Application configuration loaded from YAML."""

    def __init__(self, config_path: str = "config.yaml"):
        self._path = Path(config_path)
        self._data: dict = {}
        self.reload()

    def reload(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._data = _expand_env_vars(raw)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a value by dotted key path, e.g. 'server.port'."""
        keys = dotted_key.split(".")
        obj = self._data
        for k in keys:
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]
            else:
                return default
        return obj

    # Convenience properties
    @property
    def server_host(self) -> str:
        return self.get("server.host", "0.0.0.0")

    @property
    def server_port(self) -> int:
        return self.get("server.port", 8787)

    @property
    def embedding_model(self) -> str:
        return self.get("embedding.model_name", "BAAI/bge-large-zh-v1.5")

    @property
    def embedding_device(self) -> str:
        return self.get("embedding.device", "cuda")

    @property
    def reranker_model(self) -> str:
        return self.get("reranker.model_name", "BAAI/bge-reranker-v2-m3")

    @property
    def reranker_device(self) -> str:
        return self.get("reranker.device", "cuda")

    @property
    def vectordb_persist_dir(self) -> str:
        return self.get("vectordb.persist_dir", "./vectordb")

    @property
    def vectordb_collection(self) -> str:
        return self.get("vectordb.collection_name", "eris_scenes")

    @property
    def character_name(self) -> str:
        return self.get("character.name", "eris")

    @property
    def character_data_dir(self) -> Path:
        return Path(self.get("character.data_dir", "./data/characters/eris"))

    @property
    def novel_dir(self) -> Path:
        return Path(self.get("character.novel_dir", "./data/novel/mushoku_tensei"))

    @property
    def taxonomy_path(self) -> Path:
        return Path(self.get("taxonomy_path", "./data/tags_taxonomy.yaml"))

    @property
    def target_period(self) -> str:
        return self.get("character.target_period", "回归后")

    def get_preprocess_provider(self, stage: str) -> str:
        return self.get(f"api.preprocess.{stage}", "gemini")

    def get_runtime_provider(self, stage: str) -> str:
        return self.get(f"api.runtime.{stage}", "local")

    def get_provider_config(self, provider_name: str) -> dict:
        return self.get(f"api.providers.{provider_name}", {})

    def get_retrieval(self, key: str, default: Any = None) -> Any:
        return self.get(f"retrieval.{key}", default)
