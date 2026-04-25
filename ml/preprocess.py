"""
Preprocessing utilities.
- Loads raw CSVs from data/raw/
- Scales features using MinMaxScaler
- Creates sliding windows of length seq_len
- Saves drift baseline statistics (mean, std, min, max per feature)
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES = ["cpu_usage", "memory_usage", "request_rate", "error_rate"]
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.joblib")
BASELINE_PATH = os.path.join(PROCESSED_DIR, "drift_baseline.json")


def load_raw_data() -> pd.DataFrame:
    """Loads and concatenates all CSVs in data/raw/."""
    os.makedirs(RAW_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}. Run the Airflow DAG first.")
    dfs = []
    for f in files:
        path = os.path.join(RAW_DIR, f)
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Loaded {len(data)} rows from {len(files)} files.")
    return data


def fit_scaler(data: pd.DataFrame) -> MinMaxScaler:
    """Fits a MinMaxScaler on the feature columns and saves it."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    scaler = MinMaxScaler()
    scaler.fit(data[FEATURES])
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to {SCALER_PATH}")
    return scaler


def load_scaler() -> MinMaxScaler:
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Run preprocessing first.")
    return joblib.load(SCALER_PATH)


def save_drift_baseline(data: pd.DataFrame):
    """
    Computes and saves per-feature baseline statistics for drift detection.
    These are compared against live data during inference.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    baseline = {}
    for col in FEATURES:
        baseline[col] = {
            "mean": float(data[col].mean()),
            "std": float(data[col].std()),
            "min": float(data[col].min()),
            "max": float(data[col].max()),
            "p25": float(data[col].quantile(0.25)),
            "p75": float(data[col].quantile(0.75)),
        }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Drift baseline saved to {BASELINE_PATH}")


def make_windows(scaled: np.ndarray, seq_len: int = 60) -> np.ndarray:
    """
    Creates overlapping sliding windows.
    Input:  (N, num_features)
    Output: (N - seq_len + 1, seq_len, num_features)
    """
    windows = []
    for i in range(len(scaled) - seq_len + 1):
        windows.append(scaled[i : i + seq_len])
    return np.array(windows, dtype=np.float32)


def preprocess(seq_len: int = 60) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Load raw data
    2. Save drift baseline (on raw values)
    3. Fit & apply scaler
    4. Create sliding windows
    5. Save windows to processed/
    Returns the windows array.
    """
    data = load_raw_data()
    save_drift_baseline(data)
    scaler = fit_scaler(data)
    scaled = scaler.transform(data[FEATURES])
    windows = make_windows(scaled, seq_len)
    out_path = os.path.join(PROCESSED_DIR, "windows.npy")
    np.save(out_path, windows)
    logger.info(f"Saved {len(windows)} windows of shape {windows.shape[1:]} to {out_path}")
    return windows


def preprocess_single_window(raw_window: np.ndarray) -> np.ndarray:
    """
    Scales a single raw window for inference.
    raw_window: (seq_len, num_features) numpy array
    """
    scaler = load_scaler()
    scaled = scaler.transform(raw_window)
    return scaled.astype(np.float32)


if __name__ == "__main__":
    preprocess()
