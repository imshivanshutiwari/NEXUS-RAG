"""Pipeline checkpoint: persist and resume partial ingestion state."""

import json
import os
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_CHECKPOINT_DIR = Path("data/cache/checkpoints")


class Checkpoint:
    """JSON-backed checkpoint for resumable pipeline stages."""

    def __init__(self, name: str, directory: Optional[Path] = None) -> None:
        self.name = name
        self.directory = directory or _DEFAULT_CHECKPOINT_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path = self.directory / f"{name}.json"
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path) as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                logger.warning("Corrupt checkpoint %s – resetting.", self._path)
        return {}

    def save(self) -> None:
        with open(self._path, "w") as fh:
            json.dump(self._state, fh, indent=2)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value
        self.save()

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def mark_done(self, item_id: str) -> None:
        done = set(self._state.get("done", []))
        done.add(item_id)
        self._state["done"] = list(done)
        self.save()

    def is_done(self, item_id: str) -> bool:
        return item_id in set(self._state.get("done", []))

    def reset(self) -> None:
        self._state = {}
        if self._path.exists():
            os.remove(self._path)
