"""
Unit tests for the LSTM Autoencoder model.
Run with: pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../ml"))

import numpy as np
import torch

from model import LSTMAutoencoder
from evaluate import classify_anomaly

SEQ_LEN = 30    # NASA CMAPSS: 30 cycles per window
INPUT_SIZE = 14  # NASA CMAPSS: 14 informative sensors
BATCH = 8


def test_model_output_shape():
    model = LSTMAutoencoder(input_size=INPUT_SIZE, seq_len=SEQ_LEN)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    out = model(x)
    assert out.shape == (BATCH, SEQ_LEN, INPUT_SIZE)


def test_reconstruction_error_shape():
    model = LSTMAutoencoder(input_size=INPUT_SIZE, seq_len=SEQ_LEN)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    errors = model.reconstruction_error(x)
    assert errors.shape == (BATCH,)


def test_reconstruction_error_is_non_negative():
    model = LSTMAutoencoder(input_size=INPUT_SIZE, seq_len=SEQ_LEN)
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    errors = model.reconstruction_error(x)
    assert (errors >= 0).all()


def test_perfect_reconstruction_gives_zero_error():
    x = torch.randn(BATCH, SEQ_LEN, INPUT_SIZE)
    error = ((x - x) ** 2).mean(dim=(1, 2))
    assert torch.allclose(error, torch.zeros(BATCH))


def test_classify_normal():
    result = classify_anomaly(error=0.01, threshold=0.05)
    assert result["is_anomaly"] is False
    assert result["severity"] == "normal"


def test_classify_low_severity():
    result = classify_anomaly(error=0.07, threshold=0.05)
    assert result["is_anomaly"] is True
    assert result["severity"] == "low"


def test_classify_high_severity():
    result = classify_anomaly(error=0.15, threshold=0.05)
    assert result["is_anomaly"] is True
    assert result["severity"] == "high"


def test_classify_at_exact_threshold_is_normal():
    result = classify_anomaly(error=0.05, threshold=0.05)
    assert result["is_anomaly"] is False


def test_classify_score_is_rounded():
    result = classify_anomaly(error=0.123456789, threshold=0.05)
    assert len(str(result["score"]).split(".")[-1]) <= 6
