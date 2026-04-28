"""
Airflow DAG: ml_training_pipeline
Runs the full ML pipeline: preprocess → train → evaluate → reload model
Schedule: Daily at midnight, or triggered manually after enough data is ingested.

Tasks:
  1. check_data       — verify enough rows exist in nasa_metrics table
  2. preprocess       — run ml/preprocess.py to create windows
  3. train            — run ml/train.py to train LSTM Autoencoder
  4. evaluate         — run ml/evaluate.py to compute final threshold
  5. reload_model     — call FastAPI /reload endpoint to hot-swap model
"""

import logging
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

ML_DIR = os.environ.get("ML_DIR", "/app/ml")
DB_PATH = os.environ.get("DB_PATH", "/app/data/metrics.db")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
MIN_ROWS_REQUIRED = 500  # minimum rows before training

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}


def check_data(**context):
    """
    Checks that enough rows exist in nasa_metrics table before training.
    Fails the task if not enough data is available.
    """
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT COUNT(*) FROM nasa_metrics").fetchone()
    conn.close()
    count = row[0] if row else 0
    logger.info(f"nasa_metrics row count: {count}")
    if count < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Not enough data to train: {count} rows found, "
            f"{MIN_ROWS_REQUIRED} required."
        )
    context["ti"].xcom_push(key="row_count", value=count)
    logger.info(f"Data check passed: {count} rows available.")


def run_preprocess(**context):
    """Runs ml/preprocess.py to generate scaled windows and drift baseline."""
    result = subprocess.run(
        ["python3", os.path.join(ML_DIR, "preprocess.py")],
        capture_output=True,
        text=True,
        cwd=ML_DIR,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Preprocessing failed:\n{result.stderr}")
    logger.info("Preprocessing completed successfully.")


def run_train(**context):
    """Runs ml/train.py to train LSTM Autoencoder and register in MLflow."""
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

    result = subprocess.run(
        ["python3", os.path.join(ML_DIR, "train.py"),
         "--epochs", "50",
         "--hidden_size", "64",
         "--seq_len", "30",
         "--batch_size", "32",
         "--lr", "0.001"],
        capture_output=True,
        text=True,
        cwd=ML_DIR,
        env=env,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Training failed:\n{result.stderr}")
    logger.info("Training completed successfully.")


def run_evaluate(**context):
    """Runs ml/evaluate.py to validate threshold and log evaluation summary."""
    result = subprocess.run(
        ["python3", os.path.join(ML_DIR, "evaluate.py")],
        capture_output=True,
        text=True,
        cwd=ML_DIR,
    )
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Evaluation failed:\n{result.stderr}")
    logger.info("Evaluation completed successfully.")


def reload_backend_model(**context):
    """
    Calls FastAPI /reload endpoint to hot-swap the model without restart.
    Falls back gracefully if backend is unreachable.
    """
    import requests
    try:
        resp = requests.post(f"{BACKEND_URL}/reload", timeout=30)
        if resp.status_code == 200:
            logger.info(f"Model reloaded successfully: {resp.json()}")
        else:
            logger.warning(f"Backend reload returned {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.warning(f"Could not reach backend for reload: {e}. "
                       f"Restart backend manually to load new model.")


with DAG(
    dag_id="ml_training_pipeline",
    description="Full ML pipeline: preprocess → train → evaluate → reload",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 0 * * *",  # daily at midnight
    catchup=False,
    tags=["mlops", "training", "nasa"],
) as dag:

    t1 = PythonOperator(task_id="check_data",       python_callable=check_data)
    t2 = PythonOperator(task_id="preprocess",        python_callable=run_preprocess)
    t3 = PythonOperator(task_id="train",             python_callable=run_train)
    t4 = PythonOperator(task_id="evaluate",          python_callable=run_evaluate)
    t5 = PythonOperator(task_id="reload_model",      python_callable=reload_backend_model)

    t1 >> t2 >> t3 >> t4 >> t5
