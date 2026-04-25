"""
tests/test_services.py
=======================
Unit tests for DriftService and MetricsStore.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.services.metrics_store import MetricEntry, MetricsStore


# ── MetricsStore ──────────────────────────────────────────────────────────────

class TestMetricsStore:

    @pytest.fixture
    def store(self):
        return MetricsStore()

    def _entry(self, ts=None, is_anomaly=False, severity=0, error=0.01):
        return MetricEntry(
            timestamp=ts or datetime.utcnow(),
            cpu_percent=25.0,
            mem_percent=40.0,
            disk_io_read_mb=1.0,
            disk_io_write_mb=0.5,
            net_bytes_sent_mb=2.0,
            net_bytes_recv_mb=1.5,
            req_rate_per_sec=120.0,
            error_rate_percent=0.5,
            reconstruction_error=error,
            is_anomaly=is_anomaly,
            severity=severity,
        )

    def test_append_and_retrieve(self, store):
        e = self._entry()
        store.append(e)
        assert len(store.all()) == 1

    def test_latest_returns_n_entries(self, store):
        for _ in range(5):
            store.append(self._entry())
        latest = store.latest(3)
        assert len(latest) == 3

    def test_latest_on_empty_store(self, store):
        assert store.latest(5) == []

    def test_query_time_range(self, store):
        now = datetime.utcnow()
        old = self._entry(ts=now - timedelta(minutes=10))
        recent = self._entry(ts=now - timedelta(minutes=1))
        store.append(old)
        store.append(recent)

        results = store.query(
            from_ts=now - timedelta(minutes=2),
            to_ts=now,
        )
        assert len(results) == 1
        assert results[0].timestamp == recent.timestamp

    def test_query_empty_range(self, store):
        now = datetime.utcnow()
        store.append(self._entry(ts=now - timedelta(hours=2)))
        results = store.query(
            from_ts=now - timedelta(minutes=5),
            to_ts=now,
        )
        assert results == []

    def test_max_entries_cap(self):
        store = MetricsStore()
        store.MAX_ENTRIES = 5
        from collections import deque
        store._data = deque(maxlen=5)
        for _ in range(10):
            store.append(self._entry())
        assert len(store.all()) == 5

    def test_anomaly_entries_queryable(self, store):
        now = datetime.utcnow()
        store.append(self._entry(ts=now - timedelta(seconds=30), is_anomaly=True, severity=2))
        store.append(self._entry(ts=now - timedelta(seconds=10), is_anomaly=False, severity=0))

        entries = store.query(
            from_ts=now - timedelta(minutes=1),
            to_ts=now,
        )
        anomalies = [e for e in entries if e.is_anomaly]
        assert len(anomalies) == 1
        assert anomalies[0].severity == 2

    def test_thread_safety(self, store):
        """Multiple threads appending should not corrupt state."""
        import threading
        errors = []

        def writer():
            try:
                for _ in range(50):
                    store.append(self._entry())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"
        assert len(store.all()) <= MetricsStore.MAX_ENTRIES


# ── DriftService ──────────────────────────────────────────────────────────────

class TestDriftService:

    FEATURE_COLS = [
        "cpu_percent", "mem_percent", "disk_io_read_mb", "disk_io_write_mb",
        "net_bytes_sent_mb", "net_bytes_recv_mb", "req_rate_per_sec", "error_rate_percent",
    ]

    def _baseline(self, mean=25.0, std=5.0):
        return {
            col: {"mean": mean, "std": std, "min": 10.0, "max": 40.0,
                  "p5": 12.0, "p25": 20.0, "p75": 30.0, "p95": 38.0}
            for col in self.FEATURE_COLS
        }

    def test_no_drift_when_baselines_match(self, tmp_path):
        from app.services.drift_service import DriftService
        ref = self._baseline(mean=25.0, std=5.0)
        cur = self._baseline(mean=25.0, std=5.0)
        (tmp_path / "reference_baseline.json").write_text(json.dumps(ref))
        (tmp_path / "current_baseline.json").write_text(json.dumps(cur))

        with patch("app.services.drift_service.BASELINE_PATH", tmp_path):
            service = DriftService()
            report = service.get_drift_report()

        assert report.overall_drift_detected is False
        assert report.features_with_drift == []

    def test_drift_detected_when_mean_shifts(self, tmp_path):
        from app.services.drift_service import DriftService
        ref = self._baseline(mean=25.0, std=5.0)
        cur = self._baseline(mean=50.0, std=5.0)   # mean shifted by 5 stds — clear drift
        (tmp_path / "reference_baseline.json").write_text(json.dumps(ref))
        (tmp_path / "current_baseline.json").write_text(json.dumps(cur))

        with patch("app.services.drift_service.BASELINE_PATH", tmp_path):
            service = DriftService()
            report = service.get_drift_report()

        assert report.overall_drift_detected is True
        assert len(report.features_with_drift) > 0

    def test_report_has_all_features(self, tmp_path):
        from app.services.drift_service import DriftService
        ref = self._baseline()
        cur = self._baseline()
        (tmp_path / "reference_baseline.json").write_text(json.dumps(ref))
        (tmp_path / "current_baseline.json").write_text(json.dumps(cur))

        with patch("app.services.drift_service.BASELINE_PATH", tmp_path):
            service = DriftService()
            report = service.get_drift_report()

        assert len(report.feature_stats) == len(self.FEATURE_COLS)
        reported_features = {fs.feature for fs in report.feature_stats}
        assert reported_features == set(self.FEATURE_COLS)

    def test_psi_scores_non_negative(self, tmp_path):
        from app.services.drift_service import DriftService
        ref = self._baseline(mean=25.0, std=5.0)
        cur = self._baseline(mean=30.0, std=5.0)
        (tmp_path / "reference_baseline.json").write_text(json.dumps(ref))
        (tmp_path / "current_baseline.json").write_text(json.dumps(cur))

        with patch("app.services.drift_service.BASELINE_PATH", tmp_path):
            report = DriftService().get_drift_report()

        for fs in report.feature_stats:
            assert fs.psi_score >= 0.0

    def test_missing_baseline_returns_empty(self, tmp_path):
        from app.services.drift_service import DriftService
        # No baseline files written
        with patch("app.services.drift_service.BASELINE_PATH", tmp_path):
            report = DriftService().get_drift_report()

        # Should return a report without crashing
        assert report is not None
        assert isinstance(report.feature_stats, list)

    def test_recommendation_present(self, tmp_path):
        from app.services.drift_service import DriftService
        ref = self._baseline()
        cur = self._baseline()
        (tmp_path / "reference_baseline.json").write_text(json.dumps(ref))
        (tmp_path / "current_baseline.json").write_text(json.dumps(cur))

        with patch("app.services.drift_service.BASELINE_PATH", tmp_path):
            report = DriftService().get_drift_report()

        assert isinstance(report.recommendation, str)
        assert len(report.recommendation) > 0
