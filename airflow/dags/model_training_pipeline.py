"""
DAG: model_training_pipeline
================================
Triggered by metric_ingestion_pipeline when drift is detected or on first run.

  1. load_features        — load latest windowed .npz from features dir
  2. train_lstm           — train LSTM Autoencoder, log to MLflow
  3. evaluate_model       — compute reconstruction error stats, set threshold
  4. register_model       — push model to MLflow registry → Staging
  5. promote_model        — if eval passes, promote Staging → Production
  6. notify_deployment    — log deployment metadata
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

FEATURES_PATH = Path(os.getenv("DATA_FEATURES_PATH", "/opt/airflow/data/features"))
MODELS_PATH = Path(os.getenv("MODELS_PATH", "/opt/airflow/models/trained"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "anomaly-detection")

# Model hyperparameters (also logged to MLflow)
HPARAMS = {
    "input_dim": 8,
    "hidden_dim": 64,
    "latent_dim": 16,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 50,
    "threshold_percentile": 95,
}

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ═════════════════════════════════════════════════════════════════════════════
# Task callables
# ═════════════════════════════════════════════════════════════════════════════

def load_features(**context) -> str:
    """Load the most recent windowed feature file."""
    import numpy as np

    files = sorted(FEATURES_PATH.glob("windows_*.npz"))
    if not files:
        raise FileNotFoundError(f"No feature files found in {FEATURES_PATH}")

    latest = files[-1]
    data = np.load(latest)
    log.info("Loaded features: X=%s y=%s from %s", data["X"].shape, data["y"].shape, latest)
    return str(latest)


def train_lstm(**context) -> dict:
    """
    Train LSTM Autoencoder with MLflow experiment tracking.
    Logs: hyperparams, per-epoch train loss, final val loss, model artifact.
    Returns dict with run_id and model artifact path.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import mlflow
    import mlflow.pytorch

    features_path = context["ti"].xcom_pull(task_ids="load_features")
    data = np.load(features_path)
    X = data["X"]   # (N, window_size, n_features)

    # Train/val split (80/20) — use only normal windows for autoencoder training
    y = data["y"]
    X_normal = X[y == 0]
    split = int(len(X_normal) * 0.8)
    X_train, X_val = X_normal[:split], X_normal[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, X_train_t),
        batch_size=HPARAMS["batch_size"],
        shuffle=True,
    )

    # ── Model definition ─────────────────────────────────────────────────
    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
            super().__init__()
            self.encoder = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout
            )
            self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
            self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
            self.decoder = nn.LSTM(
                hidden_dim, input_dim, num_layers,
                batch_first=True, dropout=dropout
            )

        def forward(self, x):
            # Encode
            enc_out, (h, _) = self.encoder(x)
            latent = self.encoder_fc(h[-1])              # (batch, latent_dim)
            # Decode — repeat latent across time steps
            dec_in = self.decoder_fc(latent).unsqueeze(1).repeat(1, x.size(1), 1)
            dec_out, _ = self.decoder(dec_in)
            return dec_out

    model = LSTMAutoencoder(
        input_dim=HPARAMS["input_dim"],
        hidden_dim=HPARAMS["hidden_dim"],
        latent_dim=HPARAMS["latent_dim"],
        num_layers=HPARAMS["num_layers"],
        dropout=HPARAMS["dropout"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])
    criterion = nn.MSELoss()

    # ── MLflow run ───────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"lstm_ae_{datetime.utcnow().strftime('%Y%m%d_%H%M')}") as run:
        mlflow.log_params(HPARAMS)
        mlflow.log_param("features_file", features_path)
        mlflow.log_param("train_windows", len(X_train))
        mlflow.log_param("val_windows", len(X_val))

        # ── Training loop ─────────────────────────────────────────────────
        model.train()
        for epoch in range(HPARAMS["epochs"]):
            epoch_loss = 0.0
            for X_batch, _ in train_loader:
                optimizer.zero_grad()
                recon = model(X_batch)
                loss = criterion(recon, X_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            if (epoch + 1) % 10 == 0:
                log.info("Epoch %d/%d — train_loss=%.6f", epoch + 1, HPARAMS["epochs"], avg_loss)

        # ── Validation loss ───────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()
        mlflow.log_metric("val_loss", val_loss)
        log.info("Final val_loss=%.6f", val_loss)

        # ── Save model ────────────────────────────────────────────────────
        MODELS_PATH.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_PATH / f"lstm_ae_{run.info.run_id}.pt"
        torch.save(model.state_dict(), model_path)

        mlflow.pytorch.log_model(
            model,
            artifact_path="lstm-autoencoder",
            registered_model_name="lstm-autoencoder",
        )

        run_id = run.info.run_id
        log.info("MLflow run complete: run_id=%s", run_id)

    return {"run_id": run_id, "val_loss": val_loss, "model_path": str(model_path)}


def evaluate_model(**context) -> dict:
    """
    Compute anomaly threshold from reconstruction errors on normal data.
    Threshold = percentile(recon_errors, HPARAMS["threshold_percentile"]).
    Logs threshold to MLflow. Returns eval metadata.
    """
    import numpy as np
    import torch
    import torch.nn as nn
    import mlflow

    train_result = context["ti"].xcom_pull(task_ids="train_lstm")
    run_id = train_result["run_id"]

    features_path = context["ti"].xcom_pull(task_ids="load_features")
    data = np.load(features_path)
    X = data["X"]
    y = data["y"]
    X_normal = torch.tensor(X[y == 0], dtype=torch.float32)

    # Load model from MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/lstm-autoencoder"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    criterion = nn.MSELoss(reduction="none")
    with torch.no_grad():
        recon = model(X_normal)
        # Per-window reconstruction error: mean over time and features
        errors = criterion(recon, X_normal).mean(dim=(1, 2)).numpy()

    threshold = float(np.percentile(errors, HPARAMS["threshold_percentile"]))
    mean_err = float(errors.mean())
    std_err = float(errors.std())

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("recon_error_mean", mean_err)
        mlflow.log_metric("recon_error_std", std_err)
        mlflow.log_metric("anomaly_threshold", threshold)

    log.info(
        "Threshold set at %.6f (p%d) | mean=%.6f std=%.6f",
        threshold, HPARAMS["threshold_percentile"], mean_err, std_err
    )

    return {
        "run_id": run_id,
        "threshold": threshold,
        "recon_error_mean": mean_err,
        "recon_error_std": std_err,
        "passed": train_result["val_loss"] < 0.05,   # acceptance criterion
    }


def check_eval_passed(**context) -> str:
    """Branch: promote to Production only if acceptance criterion met."""
    eval_result = context["ti"].xcom_pull(task_ids="evaluate_model")
    if eval_result and eval_result.get("passed"):
        log.info("Eval passed — promoting model to Production")
        return "promote_model"
    log.warning("Eval failed acceptance criterion — model stays in Staging")
    return "skip_promotion"


def promote_model(**context) -> None:
    """Transition model from Staging → Production in MLflow registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    eval_result = context["ti"].xcom_pull(task_ids="evaluate_model")
    run_id = eval_result["run_id"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # Find the version registered in this run
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if not versions:
        raise RuntimeError(f"No model version found for run_id={run_id}")

    version = versions[0].version
    client.transition_model_version_stage(
        name="lstm-autoencoder",
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )

    # Tag with threshold for runtime use
    client.set_model_version_tag(
        name="lstm-autoencoder",
        version=version,
        key="anomaly_threshold",
        value=str(eval_result["threshold"]),
    )

    log.info("Model version %s promoted to Production (threshold=%.6f)", version, eval_result["threshold"])


def notify_deployment(**context) -> None:
    """Log deployment summary as MLflow run tag."""
    import mlflow
    from mlflow.tracking import MlflowClient

    eval_result = context["ti"].xcom_pull(task_ids="evaluate_model")
    if not eval_result:
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    client.set_tag(
        eval_result["run_id"],
        "deployment_ts",
        datetime.utcnow().isoformat(),
    )
    client.set_tag(eval_result["run_id"], "deployed_by", "airflow_dag")
    log.info("Deployment notification logged for run_id=%s", eval_result["run_id"])


# ═════════════════════════════════════════════════════════════════════════════
# DAG definition
# ═════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="model_training_pipeline",
    default_args=default_args,
    description="Train LSTM Autoencoder, evaluate, register and promote via MLflow",
    schedule_interval=None,   # triggered externally by ingestion DAG
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["training", "mlops", "anomaly-detection"],
) as dag:

    t_load = PythonOperator(
        task_id="load_features",
        python_callable=load_features,
    )

    t_train = PythonOperator(
        task_id="train_lstm",
        python_callable=train_lstm,
        execution_timeout=timedelta(hours=2),
    )

    t_eval = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    t_branch = BranchPythonOperator(
        task_id="check_eval_passed",
        python_callable=check_eval_passed,
    )

    t_promote = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    t_skip = EmptyOperator(task_id="skip_promotion")

    t_notify = PythonOperator(
        task_id="notify_deployment",
        python_callable=notify_deployment,
        trigger_rule="none_failed_min_one_success",
    )

    # ── DAG dependency graph ──────────────────────────────────────────────
    t_load >> t_train >> t_eval >> t_branch
    t_branch >> [t_promote, t_skip] >> t_notify
