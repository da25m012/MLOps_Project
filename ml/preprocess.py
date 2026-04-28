"""
Preprocessing utilities for NASA CMAPSS FD001 dataset.
- Loads train_FD001.txt and test_FD001.txt from data/raw/
- Selects 14 informative sensors (constant sensors dropped)
- Normalises using MinMaxScaler
- Creates sliding windows of length seq_len
- Saves drift baseline statistics
- Labels: early cycles = normal, late cycles (near failure) = anomalous
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

# Column definitions for CMAPSS dataset
COLUMN_NAMES = (
    ["engine_id", "cycle", "op1", "op2", "op3"] +
    [f"sensor{i}" for i in range(1, 22)]
)

# 14 informative sensors (drop constant ones: 1,5,6,10,16,18,19)
FEATURES = [
    "sensor2", "sensor3", "sensor4", "sensor7", "sensor8", "sensor9",
    "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
    "sensor17", "sensor20", "sensor21"
]

INPUT_SIZE = len(FEATURES)  # 14

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler.joblib")
BASELINE_PATH = os.path.join(PROCESSED_DIR, "drift_baseline.json")

# Engines are considered "normal" in first NORMAL_CYCLES cycles
NORMAL_CYCLES = 125


def load_raw_data(filename: str = "train_FD001.txt") -> pd.DataFrame:
    """Loads CMAPSS text file into a DataFrame."""
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Copy train_FD001.txt to data/raw/")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows from {filename} ({df['engine_id'].nunique()} engines)")
    return df


def fit_scaler(data: pd.DataFrame) -> MinMaxScaler:
    """Fits MinMaxScaler on feature columns and saves it."""
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
    """Saves per-feature baseline statistics for drift detection."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    baseline = {}
    for col in FEATURES:
        baseline[col] = {
            "mean": float(data[col].mean()),
            "std":  float(data[col].std()),
            "min":  float(data[col].min()),
            "max":  float(data[col].max()),
            "p25":  float(data[col].quantile(0.25)),
            "p75":  float(data[col].quantile(0.75)),
        }
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Drift baseline saved to {BASELINE_PATH}")


def make_windows(scaled: np.ndarray, seq_len: int = 30) -> np.ndarray:
    """
    Creates overlapping sliding windows.
    Input:  (N, num_features)
    Output: (N - seq_len + 1, seq_len, num_features)
    """
    windows = []
    for i in range(len(scaled) - seq_len + 1):
        windows.append(scaled[i: i + seq_len])
    return np.array(windows, dtype=np.float32)


def preprocess(seq_len: int = 30) -> np.ndarray:
    """
    Full preprocessing pipeline:
    1. Load train_FD001.txt
    2. Keep only early (normal) cycles per engine for training
    3. Save drift baseline on raw values
    4. Fit and apply MinMaxScaler
    5. Create sliding windows per engine (no cross-engine contamination)
    6. Save windows to processed/
    Returns the windows array.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    data = load_raw_data("train_FD001.txt")

    # Keep only normal (early) cycles for training the autoencoder
    normal_data = data[data["cycle"] <= NORMAL_CYCLES].copy()
    logger.info(f"Using {len(normal_data)} normal-cycle rows for training "
                f"(cycle <= {NORMAL_CYCLES})")

    save_drift_baseline(normal_data)
    scaler = fit_scaler(normal_data)

    # Create windows per engine to avoid cross-engine contamination
    all_windows = []
    for engine_id, group in normal_data.groupby("engine_id"):
        group_scaled = scaler.transform(group[FEATURES])
        if len(group_scaled) >= seq_len:
            wins = make_windows(group_scaled, seq_len)
            all_windows.append(wins)

    windows = np.concatenate(all_windows, axis=0)
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


def load_test_data() -> pd.DataFrame:
    """Loads test_FD001.txt for inference/evaluation."""
    return load_raw_data("test_FD001.txt")


if __name__ == "__main__":
    preprocess()
