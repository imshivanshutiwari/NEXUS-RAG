"""Structured logger for NEXUS-RAG pipeline."""
import logging
import sys
from datetime import datetime
from typing import Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger with the given name."""
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger


_pipeline_log: list[dict] = []


def log_pipeline_event(stage: str, message: str, duration_ms: float = 0.0) -> None:
    """Append a structured event to the in-memory pipeline log."""
    _pipeline_log.append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "stage": stage,
            "message": message,
            "duration_ms": duration_ms,
        }
    )


def get_pipeline_log() -> list[dict]:
    """Return a copy of the in-memory pipeline event log."""
    return list(_pipeline_log)


def clear_pipeline_log() -> None:
    """Clear the in-memory pipeline event log."""
    _pipeline_log.clear()
