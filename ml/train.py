"""
Training script for the LSTM Autoencoder on NASA CMAPSS FD001 dataset.
Tracks all experiments with MLflow.
"""

import argparse
import logging
import os

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import LSTMAutoencoder
from preprocess import preprocess, INPUT_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HPARAMS = {
    "input_size": INPUT_SIZE,  # 14 sensors
    "hidden_size": 64,
    "num_layers": 2,
    "seq_len": 30,             # 30 cycles per window
    "dropout": 0.2,
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 32,
    "val_split": 0.1,
    "threshold_percentile": 95,
}

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT_NAME = "nasa-cmapss-anomaly-detection"
REGISTERED_MODEL_NAME = "lstm_autoencoder"


def train(hparams=None):
    if hparams:
        HPARAMS.update(hparams)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("Running preprocessing on NASA CMAPSS FD001...")
    windows = preprocess(seq_len=HPARAMS["seq_len"])
    # FIX 9: explicit dtype=torch.float32
    tensors = torch.tensor(windows, dtype=torch.float32)

    val_size = int(len(tensors) * HPARAMS["val_split"])
    train_size = len(tensors) - val_size
    train_set, val_set = random_split(TensorDataset(tensors), [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=HPARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=HPARAMS["batch_size"])

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

    run_id = os.environ.get("MLFLOW_RUN_ID")
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_params(HPARAMS)
        logger.info(f"MLflow run ID: {run.info.run_id}")

        best_val_loss = float("inf")

        for epoch in range(1, HPARAMS["epochs"] + 1):
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

        # FIX 4: weights_only=True
        model.load_state_dict(torch.load("/tmp/best_model.pt", map_location=device, weights_only=True))
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Compute anomaly threshold on training data
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

        # Save threshold locally
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        threshold_path = os.path.join(PROCESSED_DIR, "threshold.txt")
        with open(threshold_path, "w") as f:
            f.write(str(threshold))

        # Log artifacts to MLflow
        mlflow.log_artifact(threshold_path)
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, "scaler.joblib"))
        mlflow.log_artifact(os.path.join(PROCESSED_DIR, "drift_baseline.json"))

        # Run evaluation and log report as artifact
        eval_report_path = os.path.join(PROCESSED_DIR, "eval_report.json")
        eval_summary = evaluate_dataset(model, tensors.numpy(), device)
        import json as _json
        with open(eval_report_path, "w") as _f:
            _json.dump(eval_summary, _f, indent=2)
        mlflow.log_artifact(eval_report_path)
        mlflow.log_metric("anomaly_rate", eval_summary["anomaly_rate"])
        mlflow.log_metric("anomaly_count", eval_summary["anomaly_count"])
        logger.info(f"Eval report: {eval_summary}")

        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        logger.info(f"Model registered as '{REGISTERED_MODEL_NAME}'")
        logger.info(f"Run complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", type=int, default=HPARAMS["hidden_size"])
    parser.add_argument("--num_layers", type=int, default=HPARAMS["num_layers"])
    parser.add_argument("--seq_len", type=int, default=HPARAMS["seq_len"])
    parser.add_argument("--lr", type=float, default=HPARAMS["lr"])
    parser.add_argument("--epochs", type=int, default=HPARAMS["epochs"])
    parser.add_argument("--batch_size", type=int, default=HPARAMS["batch_size"])
    args = parser.parse_args()
    train(hparams=vars(args))
