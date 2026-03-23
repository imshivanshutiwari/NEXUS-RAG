"""Monitor endpoint."""
from fastapi import APIRouter
from api.schemas import MonitorResponse
from monitoring.quality_monitor import QualityMonitor
from monitoring.latency_tracker import get_tracker

router = APIRouter()
_quality = QualityMonitor()

@router.get("/", response_model=MonitorResponse)
def get_monitor():
    means = _quality.get_rolling_means()
    stats = get_tracker().get_stage_stats()
    return MonitorResponse(
        drift_score=0.0,
        drift_status="STABLE",
        rolling_faithfulness=means.get("faithfulness", 1.0),
        rolling_relevancy=means.get("answer_relevancy", 1.0),
        latency_stats=stats,
        recent_alerts=[],
    )
