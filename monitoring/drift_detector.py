"""Evidently AI embedding drift detector."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.config_loader import ConfigLoader
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DriftReport:
    drift_score: float
    is_drift_detected: bool
    status: str  # STABLE | WATCH | DRIFT DETECTED
    details: dict = field(default_factory=dict)


class EmbeddingDriftDetector:
    """Detect embedding distribution drift using Evidently AI."""

    def __init__(self) -> None:
        try:
            cfg = ConfigLoader("monitoring_config.yaml")
            self._threshold: float = cfg.get("drift.drift_threshold", 0.3)
            self._ref_window: int = cfg.get("drift.reference_window", 1000)
            self._cur_window: int = cfg.get("drift.current_window", 100)
        except Exception:
            self._threshold = 0.3
            self._ref_window = 1000
            self._cur_window = 100

        self._reference_embeddings: Optional[np.ndarray] = None
        self._query_history: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_reference(self, embeddings: np.ndarray) -> None:
        """Set the reference embedding distribution."""
        self._reference_embeddings = embeddings
        logger.info(
            "EmbeddingDriftDetector: reference set (%d samples).", len(embeddings)
        )

    def compute_drift(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
    ) -> DriftReport:
        """Compute embedding drift between reference and current distributions."""
        try:
            from evidently.metrics import EmbeddingsDriftMetric
            from evidently.report import Report

            ref_df = pd.DataFrame(
                reference_embeddings[:, :50]
            )  # use first 50 dims for speed
            cur_df = pd.DataFrame(current_embeddings[:, :50])
            ref_df.columns = [str(c) for c in ref_df.columns]
            cur_df.columns = [str(c) for c in cur_df.columns]

            report = Report(metrics=[EmbeddingsDriftMetric()])
            report.run(reference_data=ref_df, current_data=cur_df)
            report_dict = report.as_dict()
            drift_score = self._parse_drift_score(report_dict)

        except Exception as exc:
            logger.warning(
                "Evidently drift report failed: %s — using MMD fallback.", exc
            )
            drift_score = self._mmd_drift(reference_embeddings, current_embeddings)

        status = self._status_label(drift_score)
        return DriftReport(
            drift_score=drift_score,
            is_drift_detected=drift_score >= self._threshold,
            status=status,
            details={
                "threshold": self._threshold,
                "n_ref": len(reference_embeddings),
                "n_cur": len(current_embeddings),
            },
        )

    def monitor_query_distribution(self, new_queries: List[str]) -> DriftReport:
        """Embed new queries and compare vs reference distribution."""
        if self._reference_embeddings is None:
            return DriftReport(
                drift_score=0.0, is_drift_detected=False, status="STABLE"
            )

        from ingestion.embedder import DocumentEmbedder

        embedder = DocumentEmbedder()
        current_embs = embedder.embed_documents(new_queries)
        report = self.compute_drift(self._reference_embeddings, current_embs)
        if report.is_drift_detected:
            logger.warning(
                "EmbeddingDriftDetector: DRIFT DETECTED score=%.3f", report.drift_score
            )
        return report

    def trigger_reindex_if_needed(self, drift_report: DriftReport) -> bool:
        """Trigger re-indexing if drift score exceeds threshold."""
        if drift_report.drift_score >= self._threshold:
            logger.warning(
                "EmbeddingDriftDetector: triggering re-index (drift=%.3f).",
                drift_report.drift_score,
            )
            try:
                from monitoring.self_healer import SelfHealer

                SelfHealer().reindex()
            except Exception as exc:
                logger.error("SelfHealer.reindex failed: %s", exc)
            return True
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_drift_score(report_dict: dict) -> float:
        try:
            metrics = report_dict.get("metrics", [])
            for metric in metrics:
                result = metric.get("result", {})
                score = result.get("drift_score") or result.get("score")
                if score is not None:
                    return float(score)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _mmd_drift(ref: np.ndarray, cur: np.ndarray) -> float:
        """Maximum Mean Discrepancy as drift score fallback."""
        ref_mean = ref.mean(axis=0)
        cur_mean = cur.mean(axis=0)
        diff = ref_mean - cur_mean
        return float(np.sqrt((diff**2).sum()))

    @staticmethod
    def _status_label(score: float) -> str:
        if score < 0.15:
            return "STABLE"
        if score < 0.30:
            return "WATCH"
        return "DRIFT DETECTED"
