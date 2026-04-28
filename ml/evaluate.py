"""
Evaluation utilities for NASA CMAPSS anomaly detection.
- Scores windows using reconstruction error
- Classifies anomalies as Low or High severity
- Drift detection against training baseline
"""

import json
import logging
import os

import numpy as np
import torch

from model import LSTMAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
BASELINE_PATH = os.path.join(PROCESSED_DIR, "drift_baseline.json")
THRESHOLD_PATH = os.path.join(PROCESSED_DIR, "threshold.txt")

HIGH_MULTIPLIER = 2.0


def load_threshold() -> float:
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"Threshold file not found at {THRESHOLD_PATH}.")
    with open(THRESHOLD_PATH) as f:
        return float(f.read().strip())


def score_window(model: LSTMAutoencoder, window: np.ndarray, device: torch.device) -> float:
    """
    Computes reconstruction error for a single window.
    window: (seq_len, num_features) numpy array, already scaled.
    """
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
        error = model.reconstruction_error(tensor)
    return float(error.cpu().numpy()[0])


def classify_anomaly(error: float, threshold: float) -> dict:
    """
    Returns anomaly classification dict.
    - normal: error <= threshold
    - low:    threshold < error <= HIGH_MULTIPLIER * threshold
    - high:   error > HIGH_MULTIPLIER * threshold
    """
    high_threshold = threshold * HIGH_MULTIPLIER
    if error <= threshold:
        return {"is_anomaly": False, "severity": "normal", "score": round(error, 6)}
    elif error <= high_threshold:
        return {"is_anomaly": True, "severity": "low", "score": round(error, 6)}
    else:
        return {"is_anomaly": True, "severity": "high", "score": round(error, 6)}


def detect_drift(live_stats: dict) -> dict:
    """
    Compares live feature statistics against the saved baseline.
    live_stats: {feature: {"mean": float, "std": float}}
    """
    if not os.path.exists(BASELINE_PATH):
        logger.warning("No drift baseline found. Skipping drift detection.")
        return {}

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    drift_report = {}
    for feature, stats in live_stats.items():
        if feature not in baseline:
            continue
        base = baseline[feature]
        z_score = abs(stats["mean"] - base["mean"]) / (base["std"] + 1e-9)
        drift_report[feature] = {
            "drift_detected": bool(z_score > 2.0),
            "z_score": round(float(z_score), 4),
            "baseline_mean": base["mean"],
            "live_mean": stats["mean"],
        }
    return drift_report


def evaluate_dataset(model: LSTMAutoencoder, windows: np.ndarray, device: torch.device) -> dict:
    """Runs evaluation over a full windows array."""
    threshold = load_threshold()
    errors = []
    model.eval()

    with torch.no_grad():
        tensors = torch.tensor(windows, dtype=torch.float32).to(device)
        batch_size = 64
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i: i + batch_size]
            errs = model.reconstruction_error(batch)
            errors.extend(errs.cpu().numpy())

    errors = np.array(errors)
    anomalies = errors > threshold
    high_anomalies = errors > threshold * HIGH_MULTIPLIER

    summary = {
        "total_windows": len(errors),
        "anomaly_count": int(anomalies.sum()),
        "high_severity_count": int(high_anomalies.sum()),
        "low_severity_count": int(anomalies.sum()) - int(high_anomalies.sum()),
        "anomaly_rate": round(float(anomalies.mean()), 4),
        "mean_error": round(float(errors.mean()), 6),
        "threshold": round(threshold, 6),
    }
    logger.info(f"Evaluation summary: {summary}")
    return summary


if __name__ == "__main__":
    import json
    import sys
    import torch
    import joblib

    sys.path.insert(0, os.path.dirname(__file__))
    from model import LSTMAutoencoder
    from preprocess import INPUT_SIZE, FEATURES

    WINDOWS_PATH = os.path.join(PROCESSED_DIR, "windows.npy")
    SCALER_PATH  = os.path.join(PROCESSED_DIR, "scaler.joblib")
    EVAL_PATH    = os.path.join(PROCESSED_DIR, "eval_report.json")

    if not os.path.exists(WINDOWS_PATH):
        raise FileNotFoundError(f"windows.npy not found. Run preprocess.py first.")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"threshold.txt not found. Run train.py first.")

    windows = np.load(WINDOWS_PATH).astype(np.float32)
    threshold = load_threshold()

    # Load scaler to get baseline stats for drift report
    scaler = joblib.load(SCALER_PATH)

    # Rebuild model and load from MLflow latest
    device = torch.device("cpu")
    import mlflow.pytorch
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions("lstm_autoencoder")
        if not versions:
            raise ValueError("No registered model found.")
        version = versions[0].version
        model = mlflow.pytorch.load_model(
            f"models:/lstm_autoencoder/{version}", map_location=device
        )
        model.eval()
    except Exception as e:
        logger.error(f"Could not load model from MLflow: {e}")
        sys.exit(1)

    summary = evaluate_dataset(model, windows, device)

    # Add feature drift baseline check
    baseline_path = BASELINE_PATH
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline = json.load(f)
        summary["features_monitored"] = FEATURES
        summary["baseline_features"] = list(baseline.keys())

    summary["threshold"] = threshold
    summary["seq_len"] = windows.shape[1]
    summary["num_features"] = windows.shape[2]
    summary["evaluated_at"] = __import__("datetime").datetime.utcnow().isoformat()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(EVAL_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Evaluation report saved to {EVAL_PATH}")
    print(json.dumps(summary, indent=2))
