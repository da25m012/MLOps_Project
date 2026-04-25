"""
Model loading and inference.
Loads the registered MLflow model once at startup and keeps it in memory.
"""

import logging
import os
import sys

import mlflow.pytorch
import numpy as np
import torch

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5001")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "lstm_autoencoder")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "None")   # "None" = latest version

# Paths to artifacts (mounted via Docker volume or local)
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/app/data/processed")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.joblib")
THRESHOLD_PATH = os.path.join(PROCESSED_DIR, "threshold.txt")

_model = None
_device = None
_threshold = None
_run_id = None
_scaler = None


def load_model():
    """
    Loads the latest registered MLflow model into memory.
    Called once at FastAPI startup.
    """
    global _model, _device, _threshold, _run_id, _scaler

    import joblib

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
        logger.info(f"Loading model from {model_uri}...")
        _model = mlflow.pytorch.load_model(model_uri, map_location=_device)
        _model.eval()

        # Retrieve run_id for health endpoint
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
        if versions:
            _run_id = versions[0].run_id

        logger.info(f"Model loaded. Run ID: {_run_id}")
    except Exception as e:
        logger.error(f"Failed to load MLflow model: {e}")
        _model = None

    # Load scaler
    try:
        _scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded.")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        _scaler = None

    # Load threshold
    try:
        with open(THRESHOLD_PATH) as f:
            _threshold = float(f.read().strip())
        logger.info(f"Threshold loaded: {_threshold}")
    except Exception as e:
        logger.error(f"Failed to load threshold: {e}")
        _threshold = None


def is_ready() -> bool:
    return _model is not None and _scaler is not None and _threshold is not None


def get_run_id() -> str:
    return _run_id


def get_threshold() -> float:
    return _threshold


def predict(raw_window: list) -> dict:
    """
    Runs anomaly detection on a raw (unscaled) metric window.
    raw_window: list of lists, shape (seq_len, 4).
    Returns classification dict.
    """
    from evaluate import classify_anomaly, score_window

    if not is_ready():
        raise RuntimeError("Model, scaler, or threshold not loaded.")

    arr = np.array(raw_window, dtype=np.float32)           # (seq_len, 4)
    scaled = _scaler.transform(arr)                        # (seq_len, 4)
    error = score_window(_model, scaled, _device)
    result = classify_anomaly(error, _threshold)
    result["threshold"] = round(_threshold, 6)
    return result
