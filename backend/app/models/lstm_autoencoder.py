"""
models/lstm_autoencoder.py
===========================
LSTM Autoencoder for multivariate time-series anomaly detection.

Architecture:
  Encoder: LSTM → FC → latent vector
  Decoder: FC → repeat → LSTM → reconstruction

Anomaly score  = mean MSE reconstruction error over the window.
Severity label = bucketed by multiples of the trained threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ── Severity levels ───────────────────────────────────────────────────────────
class Severity(IntEnum):
    NORMAL = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @classmethod
    def label(cls, value: int) -> str:
        return {0: "normal", 1: "low", 2: "medium", 3: "high"}.get(value, "unknown")


@dataclass
class AnomalyResult:
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    severity: int
    severity_label: str
    per_feature_errors: dict[str, float]


# ── Model definition ──────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM Autoencoder.

    Args:
        input_dim:   number of input features (default 8)
        hidden_dim:  LSTM hidden state size
        latent_dim:  bottleneck dimension
        num_layers:  stacked LSTM layers
        dropout:     dropout between LSTM layers (ignored when num_layers=1)
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=input_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → latent: (batch, latent_dim)"""
        _, (h, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(h[-1])
        return latent

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """latent: (batch, latent_dim) → recon: (batch, seq_len, input_dim)"""
        dec_in = self.decoder_fc(latent).unsqueeze(1).repeat(1, seq_len, 1)
        recon, _ = self.decoder_lstm(dec_in)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        recon = self.decode(latent, x.size(1))
        return recon


# ── Inference wrapper ─────────────────────────────────────────────────────────
class AnomalyDetector:
    """
    Wraps LSTMAutoencoder for production inference.
    Handles normalisation, windowing, thresholding, and severity scoring.
    """

    def __init__(
        self,
        model: LSTMAutoencoder,
        scaler,                          # fitted sklearn scaler
        threshold: float,
        feature_cols: list[str],
        severity_multipliers: tuple[float, float, float] = (1.5, 2.5, 4.0),
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.scaler = scaler
        self.threshold = threshold
        self.feature_cols = feature_cols
        self.severity_multipliers = severity_multipliers
        self.device = device
        self._criterion = nn.MSELoss(reduction="none")

    # ── Public API ─────────────────────────────────────────────────────────
    def predict(self, window: np.ndarray) -> AnomalyResult:
        """
        Args:
            window: numpy array of shape (window_size, n_features), raw (unscaled)

        Returns:
            AnomalyResult with reconstruction error, severity, per-feature errors
        """
        if window.shape[1] != len(self.feature_cols):
            raise ValueError(
                f"Expected {len(self.feature_cols)} features, got {window.shape[1]}"
            )

        scaled = self.scaler.transform(window).astype(np.float32)
        tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            recon = self.model(tensor)
            # Per-timestep, per-feature errors
            errors = self._criterion(recon, tensor)  # (1, seq_len, n_features)

        # Aggregate: mean over time axis → (1, n_features)
        per_feature = errors.mean(dim=1).squeeze(0).cpu().numpy()
        # Scalar score: mean over all features
        score = float(per_feature.mean())

        is_anomaly = score > self.threshold
        severity = self._severity(score)

        return AnomalyResult(
            reconstruction_error=round(score, 6),
            threshold=round(self.threshold, 6),
            is_anomaly=is_anomaly,
            severity=severity,
            severity_label=Severity.label(severity),
            per_feature_errors={
                col: round(float(err), 6)
                for col, err in zip(self.feature_cols, per_feature)
            },
        )

    def _severity(self, score: float) -> int:
        lo, med, hi = self.severity_multipliers
        if score <= self.threshold:
            return Severity.NORMAL
        elif score <= self.threshold * lo:
            return Severity.LOW
        elif score <= self.threshold * med:
            return Severity.MEDIUM
        else:
            return Severity.HIGH

    @classmethod
    def load(
        cls,
        model_path: str,
        scaler_path: str,
        threshold: float,
        feature_cols: list[str],
        model_kwargs: dict | None = None,
        **kwargs,
    ) -> "AnomalyDetector":
        """Load from disk artefacts."""
        import joblib

        kwargs_model = model_kwargs or {}
        model = LSTMAutoencoder(**kwargs_model)
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)

        scaler = joblib.load(scaler_path)
        log.info("AnomalyDetector loaded from %s (threshold=%.6f)", model_path, threshold)
        return cls(model=model, scaler=scaler, threshold=threshold,
                   feature_cols=feature_cols, **kwargs)
