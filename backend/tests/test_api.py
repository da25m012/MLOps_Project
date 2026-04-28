"""
Unit tests for the FastAPI backend.
Run with: pytest tests/ -v
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ml"))

with patch("model_loader.load_model"), patch("model_loader.is_ready", return_value=True):
    from main import app

from fastapi.testclient import TestClient
client = TestClient(app)

# 30 rows x 14 features (NASA CMAPSS dimensions)
VALID_WINDOW = [[641.0, 1589.0, 1400.0, 554.0, 2388.0, 9065.0,
                 47.0, 521.0, 2388.0, 8138.0, 8.44, 392.0, 38.8, 23.0]] * 30


def test_health_returns_200():
    with patch("main.model_loader.is_ready", return_value=True):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_degraded_when_model_not_loaded():
    with patch("main.model_loader.is_ready", return_value=False):
        resp = client.get("/health")
    assert resp.json()["status"] == "degraded"
    assert resp.json()["model_loaded"] is False


def test_ready_returns_true_when_model_loaded():
    with patch("main.model_loader.is_ready", return_value=True):
        resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_ready_returns_false_when_model_not_loaded():
    with patch("main.model_loader.is_ready", return_value=False):
        resp = client.get("/ready")
    assert resp.json()["ready"] is False


def test_predict_valid_window_normal():
    mock_result = {"is_anomaly": False, "severity": "normal", "score": 0.001, "threshold": 0.05}
    with patch("main.model_loader.is_ready", return_value=True), \
         patch("main.model_loader.predict", return_value=mock_result):
        resp = client.post("/predict", json={"window": VALID_WINDOW})
    assert resp.status_code == 200
    assert resp.json()["is_anomaly"] is False
    assert resp.json()["severity"] == "normal"


def test_predict_valid_window_anomaly():
    mock_result = {"is_anomaly": True, "severity": "high", "score": 0.15, "threshold": 0.05}
    with patch("main.model_loader.is_ready", return_value=True), \
         patch("main.model_loader.predict", return_value=mock_result):
        resp = client.post("/predict", json={"window": VALID_WINDOW})
    assert resp.status_code == 200
    assert resp.json()["severity"] == "high"


def test_predict_wrong_seq_len_returns_422():
    short_window = [[641.0, 1589.0, 1400.0, 554.0, 2388.0, 9065.0,
                     47.0, 521.0, 2388.0, 8138.0, 8.44, 392.0, 38.8, 23.0]] * 10
    with patch("main.model_loader.is_ready", return_value=True):
        resp = client.post("/predict", json={"window": short_window})
    assert resp.status_code == 422


def test_predict_wrong_feature_count_returns_422():
    bad_window = [[641.0, 1589.0]] * 30  # only 2 features instead of 14
    with patch("main.model_loader.is_ready", return_value=True):
        resp = client.post("/predict", json={"window": bad_window})
    assert resp.status_code == 422


def test_predict_model_not_ready_returns_503():
    with patch("main.model_loader.is_ready", return_value=False):
        resp = client.post("/predict", json={"window": VALID_WINDOW})
    assert resp.status_code == 503


def test_predict_with_engine_metadata():
    mock_result = {"is_anomaly": False, "severity": "normal", "score": 0.001, "threshold": 0.05}
    with patch("main.model_loader.is_ready", return_value=True), \
         patch("main.model_loader.predict", return_value=mock_result):
        resp = client.post("/predict", json={
            "window": VALID_WINDOW,
            "engine_id": 42,
            "cycle": 150
        })
    assert resp.status_code == 200
    assert resp.json()["engine_id"] == 42
    assert resp.json()["cycle"] == 150


def test_metrics_endpoint_returns_200():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert b"inference_requests_total" in resp.content
