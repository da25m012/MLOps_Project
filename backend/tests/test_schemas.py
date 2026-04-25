"""
tests/test_schemas.py
======================
Unit tests for Pydantic schema validation.
Ensures the LLD API contract is enforced correctly.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.schemas import (
    MetricWindow,
    BatchPredictRequest,
    AnomalyPrediction,
    HealthResponse,
    ReadyResponse,
)


class TestMetricWindowSchema:

    def _valid_window(self, rows=60, cols=8):
        return [[float(i % 100)] * cols for i in range(rows)]

    def test_valid_window_accepted(self):
        mw = MetricWindow(window=self._valid_window())
        assert len(mw.window) == 60

    def test_empty_window_rejected(self):
        with pytest.raises(ValidationError):
            MetricWindow(window=[])

    def test_wrong_feature_count_rejected(self):
        bad = [[1.0, 2.0, 3.0]] * 60   # 3 features
        with pytest.raises(ValidationError):
            MetricWindow(window=bad)

    def test_timestamp_optional(self):
        mw = MetricWindow(window=self._valid_window())
        assert mw.timestamp is None

    def test_timestamp_accepted(self):
        ts = datetime(2024, 6, 1, 12, 0, 0)
        mw = MetricWindow(window=self._valid_window(), timestamp=ts)
        assert mw.timestamp == ts

    def test_non_numeric_window_rejected(self):
        bad = [["a", "b", "c", "d", "e", "f", "g", "h"]] * 60
        with pytest.raises(ValidationError):
            MetricWindow(window=bad)

    def test_inconsistent_row_lengths_rejected(self):
        rows = [[1.0] * 8] * 59 + [[1.0] * 5]   # last row has only 5 features
        with pytest.raises(ValidationError):
            MetricWindow(window=rows)


class TestBatchPredictSchema:

    def _window(self):
        return [[float(i)] * 8 for i in range(60)]

    def test_single_window_batch(self):
        req = BatchPredictRequest(windows=[self._window()])
        assert len(req.windows) == 1

    def test_multi_window_batch(self):
        req = BatchPredictRequest(windows=[self._window() for _ in range(5)])
        assert len(req.windows) == 5

    def test_empty_batch_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(windows=[])

    def test_batch_too_large_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(windows=[self._window() for _ in range(200)])


class TestHealthSchema:

    def test_health_response_valid(self):
        h = HealthResponse(status="ok", version="1.0.0", timestamp=datetime.utcnow())
        assert h.status == "ok"

    def test_ready_response_valid(self):
        r = ReadyResponse(
            ready=True,
            model_loaded=True,
            model_server_reachable=True,
            mlflow_reachable=True,
        )
        assert r.ready is True
        assert r.details == {}
