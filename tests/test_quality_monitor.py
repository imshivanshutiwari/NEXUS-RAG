"""Tests for QualityMonitor and AlertManager."""

from monitoring.quality_monitor import QualityMonitor
from monitoring.alert_manager import AlertManager


def test_record_and_means():
    mon = QualityMonitor()
    mon.record(
        {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
    )
    mon.record(
        {
            "faithfulness": 0.7,
            "answer_relevancy": 0.6,
            "context_precision": 0.5,
            "context_recall": 0.4,
        }
    )
    means = mon.get_rolling_means()
    assert abs(means["faithfulness"] - 0.8) < 0.01


def test_threshold_check():
    mon = QualityMonitor()
    for _ in range(5):
        mon.record(
            {
                "faithfulness": 0.5,
                "answer_relevancy": 0.4,
                "context_precision": 0.3,
                "context_recall": 0.3,
            }
        )
    alerts = mon.check_thresholds()
    assert "faithfulness" in alerts


def test_alert_manager_faithfulness():
    mgr = AlertManager()
    alert = mgr.check_faithfulness(0.3)
    assert alert is not None
    assert alert.name == "faithfulness_low"


def test_alert_manager_no_alert():
    mgr = AlertManager()
    alert = mgr.check_faithfulness(0.95)
    assert alert is None


def test_alert_handler_called():
    calls = []
    mgr = AlertManager()
    mgr.register_handler(lambda a: calls.append(a))
    mgr.check_latency(99999)
    assert len(calls) == 1
