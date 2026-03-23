"""YAML configuration loader with dot-notation access."""
import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Load and merge YAML config files with environment variable overrides."""

    _CONFIGS_DIR = Path(__file__).parent.parent / "configs"

    def __init__(self, config_name: str) -> None:
        config_path = self._CONFIGS_DIR / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as fh:
            self._data: dict = yaml.safe_load(fh) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation key access, e.g. 'hybrid.rrf_k'."""
        parts = key.split(".")
        node = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    @property
    def data(self) -> dict:
        return self._data


def load_all_configs() -> dict[str, ConfigLoader]:
    """Load all standard config files and return as a dict."""
    names = [
        "pipeline_config.yaml",
        "model_config.yaml",
        "retrieval_config.yaml",
        "evaluation_config.yaml",
        "monitoring_config.yaml",
    ]
    return {name.replace("_config.yaml", ""): ConfigLoader(name) for name in names}
