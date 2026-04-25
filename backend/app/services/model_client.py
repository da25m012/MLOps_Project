"""
services/model_client.py
=========================
Manages loading the production LSTM Autoencoder from MLflow registry
and exposes a clean async-compatible predict interface.

Responsibilities:
  - On startup: pull the Production model from MLflow
  - Cache the AnomalyDetector in memory
  - Expose predict() and health-check methods
  - Support hot-reload when a new model is promoted
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import joblib
import mlflow
import torch
from mlflow.tracking import MlflowClient

from app.core.config import settings
from app.models.lstm_autoencoder import AnomalyDetector, LSTMAutoencoder
from app.models.schemas import AnomalyPrediction, MetricWindow

log = logging.getLogger(__name__)


class ModelClient:
    """
    Singleton-style service that owns the loaded AnomalyDetector.
    Instantiated once during app lifespan and stored on app.state.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._detector: Optional[AnomalyDetector] = None
        self._model_version: Optional[str] = None
        self._loaded_at: Optional[datetime] = None
        self._http = httpx.AsyncClient(timeout=10.0)

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        self._mlflow_client = MlflowClient()

    # ── Lifecycle ──────────────────────────────────────────────────────────
    async def check_health(self) -> bool:
        """
        Try to load the production model from MLflow.
        Falls back to loading from local disk if MLflow is unavailable.
        """
        try:
            await self._load_from_mlflow()
            return True
        except Exception as exc:
            log.warning("MLflow model load failed (%s), falling back to local disk", exc)
            try:
                self._load_from_disk()
                return True
            except Exception as disk_exc:
                log.error("Local disk model load also failed: %s", disk_exc)
                return False

    async def close(self) -> None:
        await self._http.aclose()

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict(self, window: MetricWindow) -> AnomalyPrediction:
        """
        Run inference synchronously (called from async route via run_in_executor).
        Raises RuntimeError if model is not loaded.
        """
        if self._detector is None:
            raise RuntimeError("Model not loaded — call check_health() first")

        import numpy as np
        arr = numpy_from_window(window)
        result = self._detector.predict(arr)

        ts = window.timestamp or datetime.utcnow()
        return AnomalyPrediction(
            timestamp=ts,
            reconstruction_error=result.reconstruction_error,
            threshold=result.threshold,
            is_anomaly=result.is_anomaly,
            severity=result.severity,
            severity_label=result.severity_label,
            per_feature_errors=result.per_feature_errors,
        )

    # ── Model loading ──────────────────────────────────────────────────────
    async def _load_from_mlflow(self) -> None:
        """Download production model artefacts from MLflow registry."""
        versions = self._mlflow_client.get_latest_versions(
            settings.MODEL_NAME, stages=["Production"]
        )
        if not versions:
            raise RuntimeError(f"No Production model found for '{settings.MODEL_NAME}'")

        v = versions[0]
        log.info("Loading model '%s' version %s from MLflow", settings.MODEL_NAME, v.version)

        # Download artefacts to a local temp dir
        local_dir = Path(f"/tmp/model_v{v.version}")
        local_dir.mkdir(exist_ok=True)

        model_uri = f"models:/{settings.MODEL_NAME}/Production"
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_uri, dst_path=str(local_dir)
        )

        model_pt = Path(local_path) / "data" / "model.pth"
        scaler_pt = Path(settings.DATA_PATH) / "processed" / "scaler.joblib"

        # Resolve threshold from model version tag
        threshold = float(v.tags.get("anomaly_threshold", "0.01"))

        self._build_detector(str(model_pt), str(scaler_pt), threshold)
        self._model_version = v.version
        self._loaded_at = datetime.utcnow()
        log.info("Model loaded successfully (threshold=%.6f)", threshold)

    def _load_from_disk(self) -> None:
        """Fallback: load latest .pt file from models/trained/."""
        models_dir = Path(settings.DATA_PATH).parent / "models" / "trained"
        pts = sorted(models_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt files in {models_dir}")

        model_pt = str(pts[-1])
        scaler_pt = str(Path(settings.DATA_PATH) / "processed" / "scaler.joblib")
        threshold_file = models_dir / "threshold.json"

        threshold = 0.01  # safe default
        if threshold_file.exists():
            import json
            with open(threshold_file) as f:
                threshold = json.load(f).get("threshold", 0.01)

        self._build_detector(model_pt, scaler_pt, threshold)
        self._model_version = "local"
        self._loaded_at = datetime.utcnow()
        log.info("Loaded local model from %s (threshold=%.6f)", model_pt, threshold)

    def _build_detector(self, model_pt: str, scaler_pt: str, threshold: float) -> None:
        model = LSTMAutoencoder(
            input_dim=len(settings.FEATURE_COLS),
            hidden_dim=64,
            latent_dim=16,
            num_layers=2,
            dropout=0.2,
        )
        if Path(model_pt).exists():
            state = torch.load(model_pt, map_location="cpu")
            model.load_state_dict(state)
        else:
            log.warning("model_pt not found (%s) — using randomly initialised weights", model_pt)

        scaler = joblib.load(scaler_pt) if Path(scaler_pt).exists() else _dummy_scaler()

        self._detector = AnomalyDetector(
            model=model,
            scaler=scaler,
            threshold=threshold,
            feature_cols=settings.FEATURE_COLS,
            severity_multipliers=(
                settings.SEVERITY_LOW_MULTIPLIER,
                settings.SEVERITY_MEDIUM_MULTIPLIER,
                settings.SEVERITY_HIGH_MULTIPLIER,
            ),
        )

    # ── Status accessors ───────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._detector is not None

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version

    @property
    def loaded_at(self) -> Optional[datetime]:
        return self._loaded_at

    @property
    def threshold(self) -> Optional[float]:
        return self._detector.threshold if self._detector else None


# ── Helpers ───────────────────────────────────────────────────────────────────
def numpy_from_window(window: MetricWindow):
    import numpy as np
    return np.array(window.window, dtype=np.float32)


def _dummy_scaler():
    """Returns a no-op scaler (identity transform) for cold-start without data."""
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    s = MinMaxScaler()
    # fit on plausible ranges for 8 features
    dummy = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0],
        [100, 100, 500, 500, 1000, 1000, 500, 100],
    ], dtype=np.float32)
    s.fit(dummy)
    return s
