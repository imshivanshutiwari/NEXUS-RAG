"""Per-stage latency tracker for the NEXUS-RAG pipeline."""
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_STAGES = ["router", "retriever", "reranker", "generator", "evaluator", "healer"]


@dataclass
class LatencyRecord:
    stage: str
    duration_ms: float
    query_id: str = ""
    timestamp: float = field(default_factory=time.time)


class LatencyTracker:
    """Track per-stage latency across pipeline runs."""

    def __init__(self) -> None:
        self._records: List[LatencyRecord] = []
        self._active: Dict[str, float] = {}

    def start(self, stage: str, query_id: str = "") -> None:
        """Mark the start of a pipeline stage."""
        self._active[f"{stage}:{query_id}"] = time.perf_counter()

    def stop(self, stage: str, query_id: str = "") -> float:
        """Mark the end of a pipeline stage. Returns duration in ms."""
        key = f"{stage}:{query_id}"
        start = self._active.pop(key, None)
        if start is None:
            return 0.0
        duration_ms = (time.perf_counter() - start) * 1000.0
        self._records.append(LatencyRecord(stage=stage, duration_ms=duration_ms, query_id=query_id))
        logger.debug("LatencyTracker: %s=%.1fms (query=%s)", stage, duration_ms, query_id)
        return duration_ms

    def get_stage_stats(self) -> Dict[str, Dict[str, float]]:
        """Return mean/p50/p95/max latency per stage."""
        import numpy as np

        by_stage: Dict[str, List[float]] = defaultdict(list)
        for r in self._records:
            by_stage[r.stage].append(r.duration_ms)

        stats: Dict[str, Dict[str, float]] = {}
        for stage, times in by_stage.items():
            arr = np.array(times)
            stats[stage] = {
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "max_ms": float(arr.max()),
                "count": len(arr),
            }
        return stats

    def get_latest_waterfall(self, query_id: str) -> List[Dict[str, float]]:
        """Return the latency waterfall for the most recent query."""
        records = [r for r in self._records if r.query_id == query_id]
        return [{"stage": r.stage, "duration_ms": r.duration_ms} for r in records]

    def clear(self) -> None:
        self._records.clear()
        self._active.clear()

    @property
    def all_records(self) -> List[LatencyRecord]:
        return list(self._records)


# Module-level singleton for easy import
_tracker = LatencyTracker()


def get_tracker() -> LatencyTracker:
    return _tracker
