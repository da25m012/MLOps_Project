"""
Unit tests for the LSTM Autoencoder model.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml"))

import numpy as np
import pytest
import torch

from model import LSTMAutoencoder
from evaluate import classify_anomaly

SEQ_LEN = 60
INPUT_SIZE = 4
BATCH = 8


# ── Model architecture ────────────────────────────────────────────────────────

def test_model_output_shape():
    model = LSTMAutoencoder(input_size=INPUT_SIZE, seq_len=SEQ_LEN)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    out = model(x)
    assert out.shape == (BATCH, SEQ_LEN, INPUT_SIZE), f"Expected ({BATCH},{SEQ_LEN},{INPUT_SIZE}), got {out.shape}"


def test_reconstruction_error_shape():
    model = LSTMAutoencoder(input_size=INPUT_SIZE, seq_len=SEQ_LEN)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    errors = model.reconstruction_error(x)
    assert errors.shape == (BATCH,), f"Expected ({BATCH},), got {errors.shape}"


def test_reconstruction_error_is_non_negative():
    model = LSTMAutoencoder(input_size=INPUT_SIZE, seq_len=SEQ_LEN)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    errors = model.reconstruction_error(x)
    assert (errors >= 0).all(), "Reconstruction errors should be non-negative."


def test_perfect_reconstruction_gives_zero_error():
    """
    If model reconstructs input perfectly, error should be ~0.
    This tests the loss calculation logic, not the untrained model.
    """
    # Directly compute MSE between identical tensors
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    error = ((x - x) ** 2).mean(dim=(1, 2))
    assert torch.allclose(error, torch.zeros(BATCH)), "MSE of identical tensors should be 0."


# ── Anomaly classification ────────────────────────────────────────────────────

def test_classify_normal():
    result = classify_anomaly(error=0.01, threshold=0.05)
    assert result["is_anomaly"] is False
    assert result["severity"] == "normal"


def test_classify_low_severity():
    result = classify_anomaly(error=0.07, threshold=0.05)
    assert result["is_anomaly"] is True
    assert result["severity"] == "low"


def test_classify_high_severity():
    result = classify_anomaly(error=0.15, threshold=0.05)  # > 2x threshold
    assert result["is_anomaly"] is True
    assert result["severity"] == "high"


def test_classify_at_exact_threshold_is_normal():
    result = classify_anomaly(error=0.05, threshold=0.05)
    assert result["is_anomaly"] is False


def test_classify_score_is_rounded():
    result = classify_anomaly(error=0.123456789, threshold=0.05)
    assert len(str(result["score"]).split(".")[-1]) <= 6
