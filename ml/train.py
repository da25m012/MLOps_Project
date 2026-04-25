"""
Training script for the LSTM Autoencoder.
Tracks all experiments with MLflow:
  - Parameters: hidden_size, num_layers, seq_len, lr, epochs, batch_size
  - Metrics:    train_loss per epoch, val_loss per epoch
  - Artifacts:  trained model, scaler, drift baseline, threshold
Registers the best model in the MLflow model registry.
"""

import logging
import os

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import LSTMAutoencoder
from preprocess import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Hyperparameters ──────────────────────────────────────────────────────────
HPARAMS = {
    "input_size": 4,       # CPU, memory, req_rate, error_rate
    "hidden_size": 64,
    "num_layers": 2,
    "seq_len": 60,
    "dropout": 0.2,
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    "val_split": 0.1,
    "threshold_percentile": 95,  # Reconstruction error percentile for anomaly threshold
}

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "lstm-anomaly-detection"
REGISTERED_MODEL_NAME = "lstm_autoencoder"


def train():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # ── Data ────────────────────────────────────────────────────────────────
    logger.info("Running preprocessing...")
    windows = preprocess(seq_len=HPARAMS["seq_len"])
    tensors = torch.tensor(windows)  # (N, seq_len, features)

    val_size = int(len(tensors) * HPARAMS["val_split"])
    train_size = len(tensors) - val_size
    train_set, val_set = random_split(TensorDataset(tensors), [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=HPARAMS["batch_size"])

    # ── Model ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    model = LSTMAutoencoder(
        input_size=HPARAMS["input_size"],
        hidden_size=HPARAMS["hidden_size"],
        seq_len=HPARAMS["seq_len"],
        num_layers=HPARAMS["num_layers"],
        dropout=HPARAMS["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ── MLflow run ───────────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        mlflow.log_params(HPARAMS)
        logger.info(f"MLflow run ID: {run.info.run_id}")

        best_val_loss = float("inf")

        for epoch in range(1, HPARAMS["epochs"] + 1):
            # Training
            model.train()
            train_losses = []
            for (batch,) in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(device)
                    reconstructed = model(batch)
                    loss = criterion(reconstructed, batch)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:3d}/{HPARAMS['epochs']} | train={train_loss:.6f} | val={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "/tmp/best_model.pt")

        # ── Load best weights ────────────────────────────────────────────────
        model.load_state_dict(torch.load("/tmp/best_model.pt", map_location=device))
        mlflow.log_metric("best_val_loss", best_val_loss)

        # ── Compute anomaly threshold on training data ───────────────────────
        model.eval()
        all_errors = []
        full_loader = DataLoader(TensorDataset(tensors), batch_size=HPARAMS["batch_size"])
        with torch.no_grad():
            for (batch,) in full_loader:
                batch = batch.to(device)
                errors = model.reconstruction_error(batch)
                all_errors.extend(errors.cpu().numpy())

        threshold = float(np.percentile(all_errors, HPARAMS["threshold_percentile"]))
        logger.info(f"Anomaly threshold (p{HPARAMS['threshold_percentile']}): {threshold:.6f}")
        mlflow.log_metric("anomaly_threshold", threshold)

        # Save threshold as artifact
        threshold_path = os.path.join(PROCESSED_DIR, "threshold.txt")
        with open(threshold_path, "w") as f:
            f.write(str(threshold))
        mlflow.log_artifact(threshold_path)

        # Log scaler and baseline as artifacts
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, "scaler.joblib"))
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, "drift_baseline.json"))

        # ── Log and register model ───────────────────────────────────────────
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        logger.info(f"Model registered as '{REGISTERED_MODEL_NAME}'")
        logger.info(f"Run complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()
