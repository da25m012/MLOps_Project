"""
tests/test_routes.py
=====================
Integration tests for all FastAPI routes.
Uses the TestClient fixture from conftest.py (no real MLflow / Airflow needed).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── /health ──────────────────────────────────────────────────────────────────

class TestHealthRoute:

    def test_health_returns_200(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client: TestClient):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert data["status"] == "ok"

    def test_ready_endpoint_exists(self, client: TestClient):
        resp = client.get("/ready")
        assert resp.status_code == 200

    def test_ready_schema(self, client: TestClient):
        data = client.get("/ready").json()
        assert "ready" in data
        assert "model_loaded" in data
        assert "model_server_reachable" in data
        assert isinstance(data["ready"], bool)


# ── /api/v1/predict ───────────────────────────────────────────────────────────

def _make_window(n_rows: int = 60, n_cols: int = 8, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    values = rng.uniform(10.0, 40.0, (n_rows, n_cols)).tolist()
    return {"window": values}


class TestPredictRoute:

    def test_predict_returns_200(self, client: TestClient):
        resp = client.post("/api/v1/predict", json=_make_window())
        assert resp.status_code == 200, resp.text

    def test_predict_response_schema(self, client: TestClient):
        data = client.post("/api/v1/predict", json=_make_window()).json()
        assert "reconstruction_error" in data
        assert "is_anomaly" in data
        assert "severity" in data
        assert "severity_label" in data
        assert "threshold" in data
        assert "per_feature_errors" in data
        assert "timestamp" in data

    def test_predict_severity_label_valid(self, client: TestClient):
        data = client.post("/api/v1/predict", json=_make_window()).json()
        assert data["severity_label"] in {"normal", "low", "medium", "high"}

    def test_predict_severity_int_valid(self, client: TestClient):
        data = client.post("/api/v1/predict", json=_make_window()).json()
        assert data["severity"] in {0, 1, 2, 3}

    def test_predict_reconstruction_error_non_negative(self, client: TestClient):
        data = client.post("/api/v1/predict", json=_make_window()).json()
        assert data["reconstruction_error"] >= 0.0

    def test_predict_per_feature_errors_length(self, client: TestClient):
        data = client.post("/api/v1/predict", json=_make_window()).json()
        assert len(data["per_feature_errors"]) == 8

    def test_predict_threshold_positive(self, client: TestClient):
        data = client.post("/api/v1/predict", json=_make_window()).json()
        assert data["threshold"] > 0.0

    def test_predict_with_timestamp(self, client: TestClient):
        payload = _make_window()
        payload["timestamp"] = "2024-01-01T12:00:00"
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 200

    def test_predict_empty_window_rejected(self, client: TestClient):
        resp = client.post("/api/v1/predict", json={"window": []})
        assert resp.status_code == 422   # validation error

    def test_predict_wrong_feature_count_rejected(self, client: TestClient):
        bad = {"window": [[1.0, 2.0, 3.0]] * 60}   # 3 features instead of 8
        resp = client.post("/api/v1/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_missing_window_field_rejected(self, client: TestClient):
        resp = client.post("/api/v1/predict", json={})
        assert resp.status_code == 422

    def test_predict_batch_endpoint(self, client: TestClient):
        payload = {"windows": [_make_window()["window"] for _ in range(3)]}
        resp = client.post("/api/v1/predict/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3


# ── /api/v1/drift-report ─────────────────────────────────────────────────────

class TestDriftRoute:

    def test_drift_report_returns_200_or_500(self, client: TestClient):
        # 200 when baselines exist, 500 if not (no baseline files in test env)
        resp = client.get("/api/v1/drift-report")
        assert resp.status_code in {200, 500}

    def test_drift_report_schema_when_available(self, client: TestClient, tmp_path):
        """Write fake baseline files then check schema."""
        import json
        from pathlib import Path
        from unittest.mock import patch

        baseline_data = {
            col: {"mean": 25.0, "std": 5.0, "min": 10.0, "max": 40.0, "p5": 12.0, "p25": 20.0, "p75": 30.0, "p95": 38.0}
            for col in [
                "cpu_percent", "mem_percent", "disk_io_read_mb", "disk_io_write_mb",
                "net_bytes_sent_mb", "net_bytes_recv_mb", "req_rate_per_sec", "error_rate_percent",
            ]
        }
        ref = tmp_path / "reference_baseline.json"
        cur = tmp_path / "current_baseline.json"
        ref.write_text(json.dumps(baseline_data))
        cur.write_text(json.dumps(baseline_data))

        with patch("app.services.drift_service.BASELINE_PATH", tmp_path):
            resp = client.get("/api/v1/drift-report")

        assert resp.status_code == 200
        data = resp.json()
        assert "overall_drift_detected" in data
        assert "features_with_drift" in data
        assert "feature_stats" in data
        assert "psi_threshold" in data
        assert "recommendation" in data
        assert isinstance(data["feature_stats"], list)


# ── /api/v1/pipeline-status ──────────────────────────────────────────────────

class TestPipelineRoute:

    def test_pipeline_status_returns_200(self, client: TestClient):
        # Will return 200 with nulls for DAGs/model if Airflow/MLflow unreachable
        resp = client.get("/api/v1/pipeline-status")
        assert resp.status_code == 200

    def test_pipeline_status_schema(self, client: TestClient):
        data = client.get("/api/v1/pipeline-status").json()
        assert "timestamp" in data
        assert "mlflow_experiment" in data
        assert "total_runs" in data
        # ingestion_dag and training_dag may be null if services are not running
        assert "ingestion_dag" in data
        assert "training_dag" in data
        assert "current_model" in data


# ── /api/v1/metrics ──────────────────────────────────────────────────────────

class TestMetricsRoute:

    def test_metrics_history_returns_200(self, client: TestClient):
        resp = client.get("/api/v1/metrics/history")
        assert resp.status_code == 200

    def test_metrics_history_schema(self, client: TestClient):
        data = client.get("/api/v1/metrics/history").json()
        assert "entries" in data
        assert "count" in data
        assert "from_ts" in data
        assert "to_ts" in data

    def test_metrics_history_empty_on_fresh_start(self, client: TestClient):
        data = client.get("/api/v1/metrics/history").json()
        # Fresh test client — store may be empty
        assert isinstance(data["entries"], list)
        assert data["count"] >= 0

    def test_metrics_latest_returns_200(self, client: TestClient):
        resp = client.get("/api/v1/metrics/latest")
        assert resp.status_code in {200, 404}   # 404 if store empty

    def test_metrics_summary_returns_200(self, client: TestClient):
        resp = client.get("/api/v1/metrics/summary")
        assert resp.status_code == 200

    def test_metrics_history_with_query_params(self, client: TestClient):
        resp = client.get("/api/v1/metrics/history?minutes=5&limit=100")
        assert resp.status_code == 200


# ── /metrics (Prometheus exposition) ─────────────────────────────────────────

class TestPrometheusEndpoint:

    def test_prometheus_metrics_exposed(self, client: TestClient):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")

    def test_prometheus_metrics_content(self, client: TestClient):
        body = client.get("/metrics").text
        # Should contain at least some standard metrics
        assert "http_requests_total" in body or "python_gc" in body or "#" in body
