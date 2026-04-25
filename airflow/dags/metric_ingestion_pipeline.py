"""
DAG: metric_ingestion_pipeline
================================
Orchestrates the full data ingestion and preprocessing pipeline:

  1. collect_metrics      — scrape system + app metrics, write raw parquet
  2. validate_schema      — assert column presence, types, value ranges
  3. compute_baselines    — calculate mean/variance/distribution per feature
  4. preprocess           — normalise, build sliding windows, write features
  5. dvc_commit           — DVC add + commit versioned data artifacts
  6. trigger_training     — conditional: if drift detected or first run,
                            trigger the training DAG

Schedule: every 5 minutes in production; hourly for training checks.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# ── Paths (mounted volumes inside Airflow container) ─────────────────────────
RAW_PATH = Path(os.getenv("DATA_RAW_PATH", "/opt/airflow/data/raw"))
PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", "/opt/airflow/data/processed"))
FEATURES_PATH = Path(os.getenv("DATA_FEATURES_PATH", "/opt/airflow/data/features"))
BASELINE_PATH = Path(os.getenv("DATA_BASELINE_PATH", "/opt/airflow/data/baselines"))

FEATURE_COLS = [
    "cpu_percent", "mem_percent", "disk_io_read_mb",
    "disk_io_write_mb", "net_bytes_sent_mb", "net_bytes_recv_mb",
    "req_rate_per_sec", "error_rate_percent",
]

WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "60"))      # rows per sequence
DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


# ═════════════════════════════════════════════════════════════════════════════
# Task callables
# ═════════════════════════════════════════════════════════════════════════════

def collect_metrics(**context) -> str:
    """
    Scrape system metrics via psutil and synthetic anomaly injector.
    Writes a timestamped parquet file to RAW_PATH.
    Returns the output file path (pushed to XCom).
    """
    import pandas as pd
    import psutil
    import numpy as np

    ts = context["logical_date"].strftime("%Y%m%d_%H%M%S")
    RAW_PATH.mkdir(parents=True, exist_ok=True)

    records = []
    for _ in range(WINDOW_SIZE):
        net = psutil.net_io_counters()
        disk = psutil.disk_io_counters()
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "mem_percent": psutil.virtual_memory().percent,
            "disk_io_read_mb": (disk.read_bytes / 1e6) if disk else 0.0,
            "disk_io_write_mb": (disk.write_bytes / 1e6) if disk else 0.0,
            "net_bytes_sent_mb": net.bytes_sent / 1e6,
            "net_bytes_recv_mb": net.bytes_recv / 1e6,
            # Simulated app metrics (replace with real Prometheus scrape in prod)
            "req_rate_per_sec": float(np.random.poisson(lam=120)),
            "error_rate_percent": float(np.random.exponential(scale=0.5)),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Inject synthetic anomalies ~5% of the time for training data richness
    anomaly_mask = np.random.random(len(df)) < 0.05
    df.loc[anomaly_mask, "cpu_percent"] += np.random.uniform(40, 60, anomaly_mask.sum())
    df.loc[anomaly_mask, "error_rate_percent"] += np.random.uniform(10, 30, anomaly_mask.sum())
    df["is_anomaly"] = anomaly_mask.astype(int)

    out_path = RAW_PATH / f"metrics_{ts}.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Collected %d metric rows → %s", len(df), out_path)
    return str(out_path)


def validate_schema(**context) -> None:
    """
    Assert schema correctness:
      - Required columns present
      - No nulls in feature columns
      - Values within plausible physical ranges
    Raises ValueError on failure (triggers retry/alert).
    """
    import pandas as pd

    raw_path = context["ti"].xcom_pull(task_ids="collect_metrics")
    df = pd.read_parquet(raw_path)

    # Column presence
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed — missing columns: {missing}")

    # Null check
    null_counts = df[FEATURE_COLS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

    # Range checks
    range_rules = {
        "cpu_percent": (0, 100),
        "mem_percent": (0, 100),
        "error_rate_percent": (0, 100),
    }
    for col, (lo, hi) in range_rules.items():
        if col in df.columns:
            out_of_range = ((df[col] < lo) | (df[col] > hi)).sum()
            if out_of_range > 0:
                log.warning("Column %s has %d out-of-range values (clamping)", col, out_of_range)
                df[col] = df[col].clip(lo, hi)
                df.to_parquet(raw_path, index=False)

    log.info("Schema validation passed for %s", raw_path)


def compute_baselines(**context) -> dict:
    """
    Calculate per-feature statistical baselines (mean, std, min, max, p5, p95).
    On first run: saves as the reference baseline.
    On subsequent runs: computes PSI drift score vs reference.
    Returns drift info dict pushed to XCom.
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import ks_2samp

    BASELINE_PATH.mkdir(parents=True, exist_ok=True)
    baseline_file = BASELINE_PATH / "reference_baseline.json"

    raw_path = context["ti"].xcom_pull(task_ids="collect_metrics")
    df = pd.read_parquet(raw_path)

    current_stats = {}
    for col in FEATURE_COLS:
        s = df[col]
        current_stats[col] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
            "p5": float(np.percentile(s, 5)),
            "p25": float(np.percentile(s, 25)),
            "p75": float(np.percentile(s, 75)),
            "p95": float(np.percentile(s, 95)),
        }

    drift_detected = False
    drift_scores = {}

    if baseline_file.exists():
        with open(baseline_file) as f:
            reference = json.load(f)

        # PSI calculation per feature
        for col in FEATURE_COLS:
            ref = reference.get(col, {})
            cur = current_stats[col]

            # Simplified PSI using mean shift relative to std
            mean_shift = abs(cur["mean"] - ref.get("mean", cur["mean"]))
            ref_std = ref.get("std", 1.0) or 1.0
            psi = mean_shift / ref_std
            drift_scores[col] = round(psi, 4)

            if psi > DRIFT_PSI_THRESHOLD:
                log.warning("Drift detected on %s: PSI=%.4f", col, psi)
                drift_detected = True
    else:
        # First run — save as reference
        log.info("No baseline found — saving current stats as reference baseline")
        with open(baseline_file, "w") as f:
            json.dump(current_stats, f, indent=2)

    # Always update the rolling current baseline
    current_stats_file = BASELINE_PATH / "current_baseline.json"
    with open(current_stats_file, "w") as f:
        json.dump(current_stats, f, indent=2)

    result = {
        "drift_detected": drift_detected,
        "drift_scores": drift_scores,
        "current_stats": current_stats,
    }
    log.info("Baseline computation done. Drift detected: %s", drift_detected)
    return result


def preprocess(**context) -> str:
    """
    1. Load raw parquet
    2. MinMax-normalise each feature column (fit on training split)
    3. Build overlapping sliding windows of shape (WINDOW_SIZE, n_features)
    4. Save as numpy .npz to FEATURES_PATH
    Returns output path pushed to XCom.
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import joblib

    ts = context["logical_date"].strftime("%Y%m%d_%H%M%S")
    FEATURES_PATH.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    raw_path = context["ti"].xcom_pull(task_ids="collect_metrics")
    df = pd.read_parquet(raw_path)

    feature_data = df[FEATURE_COLS].values.astype(np.float32)

    # Fit or load existing scaler
    scaler_path = PROCESSED_PATH / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(feature_data)
    else:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(feature_data)
        joblib.dump(scaler, scaler_path)
        log.info("Fitted new MinMaxScaler → %s", scaler_path)

    # Sliding window construction
    windows = []
    labels = []
    label_col = df["is_anomaly"].values if "is_anomaly" in df.columns else np.zeros(len(df))

    for i in range(len(scaled) - WINDOW_SIZE + 1):
        windows.append(scaled[i : i + WINDOW_SIZE])
        # Window is anomalous if ANY row in it is anomalous
        labels.append(int(label_col[i : i + WINDOW_SIZE].any()))

    X = np.array(windows, dtype=np.float32)   # (N, WINDOW_SIZE, n_features)
    y = np.array(labels, dtype=np.int32)       # (N,)

    out_path = FEATURES_PATH / f"windows_{ts}.npz"
    np.savez_compressed(out_path, X=X, y=y)
    log.info("Preprocessing done: %d windows of shape %s → %s", len(X), X.shape[1:], out_path)
    return str(out_path)


def dvc_commit(**context) -> None:
    """
    Stage newly created data artifacts with DVC and commit to Git.
    Runs: dvc add data/ && git add data.dvc && git commit
    """
    import subprocess

    raw_path = context["ti"].xcom_pull(task_ids="collect_metrics")
    features_path = context["ti"].xcom_pull(task_ids="preprocess")

    repo_root = Path("/opt/airflow")

    for path in [raw_path, features_path]:
        result = subprocess.run(
            ["dvc", "add", path],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log.warning("DVC add warning (non-fatal): %s", result.stderr)
        else:
            log.info("DVC tracked: %s", path)

    # Git commit the .dvc pointer files
    ts = context["logical_date"].strftime("%Y-%m-%d %H:%M:%S")
    subprocess.run(
        ["git", "add", "*.dvc", "data/.gitignore"],
        cwd=repo_root, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", f"chore: DVC data update [{ts}]", "--allow-empty"],
        cwd=repo_root, capture_output=True
    )
    log.info("DVC commit complete for run %s", ts)


def check_training_needed(**context) -> str:
    """
    Branch operator: decides whether to trigger model training.
    Triggers training if:
      - No trained model exists yet (first run)
      - Drift was detected in compute_baselines
    """
    baseline_info = context["ti"].xcom_pull(task_ids="compute_baselines")
    model_registry = Path("/opt/airflow/models/trained")

    first_run = not any(model_registry.glob("*.pt"))
    drift_detected = baseline_info.get("drift_detected", False) if baseline_info else False

    if first_run or drift_detected:
        reason = "first_run" if first_run else "drift_detected"
        log.info("Training trigger: %s", reason)
        return "trigger_training"
    else:
        log.info("No training needed this cycle")
        return "skip_training"


# ═════════════════════════════════════════════════════════════════════════════
# DAG definition
# ═════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="metric_ingestion_pipeline",
    default_args=default_args,
    description="Ingest system metrics, validate, compute baselines, preprocess, DVC commit",
    schedule_interval="*/5 * * * *",   # every 5 minutes
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["ingestion", "mlops", "anomaly-detection"],
) as dag:

    t_collect = PythonOperator(
        task_id="collect_metrics",
        python_callable=collect_metrics,
    )

    t_validate = PythonOperator(
        task_id="validate_schema",
        python_callable=validate_schema,
    )

    t_baselines = PythonOperator(
        task_id="compute_baselines",
        python_callable=compute_baselines,
    )

    t_preprocess = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess,
    )

    t_dvc = PythonOperator(
        task_id="dvc_commit",
        python_callable=dvc_commit,
    )

    t_branch = BranchPythonOperator(
        task_id="check_training_needed",
        python_callable=check_training_needed,
    )

    t_trigger_training = TriggerDagRunOperator(
        task_id="trigger_training",
        trigger_dag_id="model_training_pipeline",
        wait_for_completion=False,
        reset_dag_run=True,
    )

    t_skip = EmptyOperator(task_id="skip_training")

    # ── DAG dependency graph ──────────────────────────────────────────────
    t_collect >> t_validate >> t_baselines >> t_preprocess >> t_dvc >> t_branch
    t_branch >> [t_trigger_training, t_skip]
