"""
conftest.py — shared pytest fixtures
"""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import create_app
from app.models.lstm_autoencoder import AnomalyDetector, LSTMAutoencoder
from app.services.model_client import ModelClient, _dummy_scaler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def feature_cols():
    return settings.FEATURE_COLS


@pytest.fixture(scope="session")
def lstm_model(feature_cols):
    """Randomly initialised LSTM model (no disk I/O needed)."""
    return LSTMAutoencoder(
        input_dim=len(feature_cols),
        hidden_dim=32,
        latent_dim=8,
        num_layers=1,
        dropout=0.0,
    )


@pytest.fixture(scope="session")
def scaler():
    return _dummy_scaler()


@pytest.fixture(scope="session")
def detector(lstm_model, scaler, feature_cols):
    return AnomalyDetector(
        model=lstm_model,
        scaler=scaler,
        threshold=0.05,
        feature_cols=feature_cols,
        severity_multipliers=(1.5, 2.5, 4.0),
    )


@pytest.fixture(scope="session")
def normal_window(feature_cols):
    """A window of normal-looking metric values (all near zero after scaling)."""
    rng = np.random.default_rng(42)
    window = rng.uniform(low=10.0, high=30.0, size=(60, len(feature_cols))).astype(np.float32)
    return window


@pytest.fixture(scope="session")
def anomalous_window(feature_cols):
    """A window with spike values that should produce high reconstruction error."""
    rng = np.random.default_rng(99)
    window = rng.uniform(low=10.0, high=30.0, size=(60, len(feature_cols))).astype(np.float32)
    # Inject anomaly: CPU spike to 99%
    window[30:40, 0] = 99.0
    window[30:40, 7] = 80.0   # error rate spike
    return window


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient with a mocked model pre-loaded."""
    app = create_app()

    # Inject a real (random) detector so no MLflow call needed
    mock_client = _MockModelClient()
    app.state.model_client = mock_client

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class _MockModelClient:
    """Minimal stand-in for ModelClient for route-level tests."""

    def __init__(self):
        from app.models.lstm_autoencoder import AnomalyDetector, LSTMAutoencoder
        model = LSTMAutoencoder(input_dim=8, hidden_dim=32, latent_dim=8, num_layers=1, dropout=0.0)
        self._detector = AnomalyDetector(
            model=model,
            scaler=_dummy_scaler(),
            threshold=0.05,
            feature_cols=settings.FEATURE_COLS,
        )
        self._version = "test-v1"
        self._loaded_at = __import__("datetime").datetime.utcnow()

    @property
    def is_loaded(self): return True

    @property
    def model_version(self): return self._version

    @property
    def loaded_at(self): return self._loaded_at

    @property
    def threshold(self): return self._detector.threshold

    def predict(self, window):
        import numpy as np
        from app.models.schemas import AnomalyPrediction
        import datetime
        arr = np.array(window.window, dtype=np.float32)
        result = self._detector.predict(arr)
        return AnomalyPrediction(
            timestamp=window.timestamp or datetime.datetime.utcnow(),
            reconstruction_error=result.reconstruction_error,
            threshold=result.threshold,
            is_anomaly=result.is_anomaly,
            severity=result.severity,
            severity_label=result.severity_label,
            per_feature_errors=result.per_feature_errors,
        )

    async def check_health(self): return True
    async def close(self): pass
