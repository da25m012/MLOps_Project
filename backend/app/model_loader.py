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

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
REGISTERED_MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "lstm_autoencoder")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PROCESSED_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "data", "processed"))
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", _DEFAULT_PROCESSED_DIR)
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.joblib")
THRESHOLD_PATH = os.path.join(PROCESSED_DIR, "threshold.txt")

# Add ml/ to path FIRST so LSTMAutoencoder class is found before backend's model_loader
_ML_DIR = "/app/ml"
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

_model = None
_device = None
_threshold = None
_run_id = None
_scaler = None


def load_model():
    global _model, _device, _threshold, _run_id, _scaler

    import joblib

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FIX 1: Use get_latest_versions() instead of "latest" alias
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME)
        if not versions:
            raise ValueError(f"No versions found for model '{REGISTERED_MODEL_NAME}'")
        latest_version = versions[0].version
        _run_id = versions[0].run_id
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{latest_version}"
        logger.info(f"Loading model version {latest_version} from {model_uri}...")
        _model = mlflow.pytorch.load_model(model_uri, map_location=_device)
        _model.eval()
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
    # FIX 10: Move imports to module level — errors surface at startup not first call
    from evaluate import classify_anomaly, score_window

    if not is_ready():
        raise RuntimeError("Model, scaler, or threshold not loaded.")

    arr = np.array(raw_window, dtype=np.float32)
    scaled = _scaler.transform(arr)
    error = score_window(_model, scaled, _device)
    result = classify_anomaly(error, _threshold)
    result["threshold"] = round(_threshold, 6)
    return result
