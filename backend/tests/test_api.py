"""
Unit tests for the FastAPI backend.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Patch model loading before import
with patch("model.load_model"), patch("model.is_ready", return_value=True):
    from app.main import app

client = TestClient(app)

VALID_WINDOW = [[0.45, 0.60, 120.0, 0.01]] * 60   # 60 rows × 4 features


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_returns_200():
    with patch("app.main.model_loader.is_ready", return_value=True):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_degraded_when_model_not_loaded():
    with patch("app.main.model_loader.is_ready", return_value=False):
        resp = client.get("/health")
    assert resp.json()["status"] == "degraded"
    assert resp.json()["model_loaded"] is False


# ── /ready ────────────────────────────────────────────────────────────────────

def test_ready_returns_true_when_model_loaded():
    with patch("app.main.model_loader.is_ready", return_value=True):
        resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_ready_returns_false_when_model_not_loaded():
    with patch("app.main.model_loader.is_ready", return_value=False):
        resp = client.get("/ready")
    assert resp.json()["ready"] is False


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_valid_window_normal():
    mock_result = {"is_anomaly": False, "severity": "normal", "score": 0.001, "threshold": 0.05}
    with patch("app.main.model_loader.is_ready", return_value=True), \
         patch("app.main.model_loader.predict", return_value=mock_result):
        resp = client.post("/predict", json={"window": VALID_WINDOW})
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_anomaly"] is False
    assert data["severity"] == "normal"


def test_predict_valid_window_anomaly():
    mock_result = {"is_anomaly": True, "severity": "high", "score": 0.15, "threshold": 0.05}
    with patch("app.main.model_loader.is_ready", return_value=True), \
         patch("app.main.model_loader.predict", return_value=mock_result):
        resp = client.post("/predict", json={"window": VALID_WINDOW})
    assert resp.status_code == 200
    assert resp.json()["severity"] == "high"


def test_predict_wrong_seq_len_returns_422():
    short_window = [[0.5, 0.5, 100.0, 0.01]] * 10  # wrong seq_len
    with patch("app.main.model_loader.is_ready", return_value=True):
        resp = client.post("/predict", json={"window": short_window})
    assert resp.status_code == 422


def test_predict_wrong_feature_count_returns_422():
    bad_window = [[0.5, 0.5]] * 60  # only 2 features instead of 4
    with patch("app.main.model_loader.is_ready", return_value=True):
        resp = client.post("/predict", json={"window": bad_window})
    assert resp.status_code == 422


def test_predict_model_not_ready_returns_503():
    with patch("app.main.model_loader.is_ready", return_value=False):
        resp = client.post("/predict", json={"window": VALID_WINDOW})
    assert resp.status_code == 503


# ── /metrics ──────────────────────────────────────────────────────────────────

def test_metrics_endpoint_returns_200():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"inference_requests_total" in resp.content
