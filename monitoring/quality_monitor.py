"""Quality monitor: track RAGAS score history and surface trends."""

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


class QualityMonitor:
    """Monitor RAGAS score history and detect quality degradation."""

    def __init__(self, window_size: int = 100) -> None:
        self._window: Deque[Dict[str, float]] = deque(maxlen=window_size)
        try:
            cfg = ConfigLoader("evaluation_config.yaml")
            self._thresholds = {
                "faithfulness": cfg.get("ragas.faithfulness_threshold", 0.7),
                "answer_relevancy": cfg.get("ragas.relevancy_threshold", 0.6),
                "context_precision": cfg.get("ragas.precision_threshold", 0.5),
                "context_recall": cfg.get("ragas.recall_threshold", 0.5),
            }
        except Exception:
            self._thresholds = {
                "faithfulness": 0.7,
                "answer_relevancy": 0.6,
                "context_precision": 0.5,
                "context_recall": 0.5,
            }

    def record(self, scores: Dict[str, float]) -> None:
        """Add a new set of RAGAS scores to the history."""
        self._window.append(scores)

    def get_rolling_means(self) -> Dict[str, float]:
        """Return rolling mean for each metric."""
        if not self._window:
            return {}
        keys = list(self._window)[0].keys()
        return {k: float(np.mean([s.get(k, 0.0) for s in self._window])) for k in keys}

    def get_trend(self, metric: str, n: int = 20) -> str:
        """Return 'improving', 'degrading', or 'stable' trend for *metric*."""
        history = [s.get(metric, 0.0) for s in list(self._window)[-n:]]
        if len(history) < 4:
            return "stable"
        first_half = np.mean(history[: len(history) // 2])
        second_half = np.mean(history[len(history) // 2 :])
        delta = second_half - first_half
        if delta > 0.03:
            return "improving"
        if delta < -0.03:
            return "degrading"
        return "stable"

    def check_thresholds(self) -> Dict[str, Any]:
        """Return which metrics are below their thresholds."""
        means = self.get_rolling_means()
        alerts = {}
        for metric, threshold in self._thresholds.items():
            current = means.get(metric, 1.0)
            if current < threshold:
                alerts[metric] = {
                    "current": current,
                    "threshold": threshold,
                    "status": "BELOW_THRESHOLD",
                }
        return alerts

    def get_history(self) -> List[Dict[str, float]]:
        return list(self._window)
