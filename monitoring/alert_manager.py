"""Threshold-based alert manager for the NEXUS-RAG monitoring system."""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Alert:
    name: str
    severity: str  # INFO | WARNING | CRITICAL
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


class AlertManager:
    """Fire, log and optionally dispatch alerts based on metric thresholds."""

    def __init__(self) -> None:
        try:
            cfg = ConfigLoader("monitoring_config.yaml")
            self._thresholds = {
                "faithfulness_low": cfg.get("alerts.faithfulness_low", 0.6),
                "latency_high_ms": cfg.get("alerts.latency_high_ms", 3000),
                "drift_detected": cfg.get("alerts.drift_detected", 0.3),
            }
        except Exception:
            self._thresholds = {
                "faithfulness_low": 0.6,
                "latency_high_ms": 3000,
                "drift_detected": 0.3,
            }
        self._alert_history: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a callback to be called when an alert fires."""
        self._handlers.append(handler)

    def check_faithfulness(self, score: float) -> Optional[Alert]:
        """Fire alert if faithfulness is below threshold."""
        threshold = self._thresholds["faithfulness_low"]
        if score < threshold:
            return self._fire(
                Alert(
                    name="faithfulness_low",
                    severity="WARNING",
                    message=f"Faithfulness {score:.2f} < threshold {threshold:.2f}",
                    value=score,
                    threshold=threshold,
                )
            )
        return None

    def check_latency(self, latency_ms: float) -> Optional[Alert]:
        """Fire alert if latency exceeds threshold."""
        threshold = self._thresholds["latency_high_ms"]
        if latency_ms > threshold:
            return self._fire(
                Alert(
                    name="latency_high",
                    severity="WARNING",
                    message=f"Latency {latency_ms:.0f}ms > threshold {threshold:.0f}ms",
                    value=latency_ms,
                    threshold=threshold,
                )
            )
        return None

    def check_drift(self, drift_score: float) -> Optional[Alert]:
        """Fire alert if embedding drift exceeds threshold."""
        threshold = self._thresholds["drift_detected"]
        if drift_score >= threshold:
            return self._fire(
                Alert(
                    name="drift_detected",
                    severity="CRITICAL",
                    message=f"Embedding drift {drift_score:.3f} >= threshold {threshold:.2f}",
                    value=drift_score,
                    threshold=threshold,
                )
            )
        return None

    def get_recent_alerts(self, n: int = 50) -> List[Alert]:
        return self._alert_history[-n:]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fire(self, alert: Alert) -> Alert:
        self._alert_history.append(alert)
        logger.warning("ALERT [%s] %s: %s", alert.severity, alert.name, alert.message)
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as exc:
                logger.error("AlertManager handler error: %s", exc)
        return alert
