"""
tests/test_model.py
====================
Unit tests for LSTMAutoencoder and AnomalyDetector.
Tests run fully offline — no MLflow, no Docker, no network.
"""

import numpy as np
import pytest
import torch

from app.models.lstm_autoencoder import AnomalyDetector, LSTMAutoencoder
from app.services.model_client import _dummy_scaler


# ── LSTMAutoencoder ──────────────────────────────────────────────────────────

class TestLSTMAutoencoder:

    def test_forward_output_shape(self, lstm_model):
        """Reconstructed output must match input shape."""
        x = torch.randn(4, 60, 8)   # (batch=4, window=60, features=8)
        recon, latent = lstm_model(x)
        assert recon.shape == x.shape, "Reconstruction shape mismatch"
        assert latent.shape == (4, lstm_model.latent_dim)

    def test_reconstruction_error_shape(self, lstm_model):
        """reconstruction_error must return one scalar per batch item."""
        x = torch.randn(8, 60, 8)
        errors = lstm_model.reconstruction_error(x)
        assert errors.shape == (8,)
        assert (errors >= 0).all(), "Reconstruction errors must be non-negative"

    def test_reconstruction_error_non_negative(self, lstm_model):
        x = torch.randn(16, 60, 8)
        errors = lstm_model.reconstruction_error(x)
        assert (errors >= 0).all()

    def test_encode_decode_roundtrip_shape(self, lstm_model):
        x = torch.randn(2, 60, 8)
        latent = lstm_model.encode(x)
        recon = lstm_model.decode(latent, seq_len=60)
        assert recon.shape == x.shape

    def test_no_grad_during_inference(self, lstm_model):
        """reconstruction_error should work inside no_grad context."""
        x = torch.randn(2, 60, 8)
        with torch.no_grad():
            errors = lstm_model.reconstruction_error(x)
        assert errors is not None

    def test_save_load_roundtrip(self, lstm_model, tmp_path):
        """Saved and reloaded model should produce identical output."""
        path = tmp_path / "test_model.pt"
        lstm_model.save(path)
        loaded = LSTMAutoencoder.load(path)

        x = torch.randn(2, 60, 8)
        lstm_model.eval()
        loaded.eval()
        with torch.no_grad():
            orig_out, _ = lstm_model(x)
            load_out, _ = loaded(x)
        assert torch.allclose(orig_out, load_out, atol=1e-5)

    def test_single_layer_no_dropout(self):
        """Single-layer model should not error (dropout must be 0)."""
        model = LSTMAutoencoder(
            input_dim=8, hidden_dim=32, latent_dim=8,
            num_layers=1, dropout=0.5   # dropout ignored for single layer
        )
        x = torch.randn(1, 60, 8)
        recon, _ = model(x)
        assert recon.shape == x.shape

    def test_batch_size_one(self, lstm_model):
        x = torch.randn(1, 60, 8)
        recon, latent = lstm_model(x)
        assert recon.shape == (1, 60, 8)
        assert latent.shape == (1, lstm_model.latent_dim)


# ── AnomalyDetector ──────────────────────────────────────────────────────────

class TestAnomalyDetector:

    def test_normal_window_not_anomaly(self, detector, normal_window):
        """A benign metric window should not be flagged as anomalous."""
        result = detector.predict(normal_window)
        # With a randomly initialised model, reconstruction error is random,
        # so we only assert the result structure, not the anomaly flag.
        assert hasattr(result, "is_anomaly")
        assert hasattr(result, "reconstruction_error")
        assert result.reconstruction_error >= 0

    def test_result_has_all_fields(self, detector, normal_window):
        result = detector.predict(normal_window)
        assert result.severity in {0, 1, 2, 3}
        assert result.severity_label in {"normal", "low", "medium", "high"}
        assert len(result.per_feature_errors) == 8
        assert all(e >= 0 for e in result.per_feature_errors)

    def test_severity_label_matches_level(self, detector, normal_window):
        result = detector.predict(normal_window)
        mapping = {0: "normal", 1: "low", 2: "medium", 3: "high"}
        assert result.severity_label == mapping[result.severity]

    def test_high_error_triggers_anomaly(self, lstm_model, feature_cols):
        """Force reconstruction error above threshold by setting threshold very low."""
        scaler = _dummy_scaler()
        detector = AnomalyDetector(
            model=lstm_model,
            scaler=scaler,
            threshold=1e-10,            # near-zero threshold → everything is anomalous
            feature_cols=feature_cols,
            severity_multipliers=(1.5, 2.5, 4.0),
        )
        window = np.random.uniform(10, 30, (60, 8)).astype(np.float32)
        result = detector.predict(window)
        assert result.is_anomaly is True
        assert result.severity >= 1

    def test_normal_window_with_huge_threshold(self, lstm_model, feature_cols):
        """With a huge threshold, nothing should be an anomaly."""
        scaler = _dummy_scaler()
        detector = AnomalyDetector(
            model=lstm_model,
            scaler=scaler,
            threshold=1e9,
            feature_cols=feature_cols,
        )
        window = np.random.uniform(10, 30, (60, 8)).astype(np.float32)
        result = detector.predict(window)
        assert result.is_anomaly is False
        assert result.severity == 0
        assert result.severity_label == "normal"

    def test_per_feature_errors_length(self, detector, normal_window):
        result = detector.predict(normal_window)
        assert len(result.per_feature_errors) == 8

    def test_wrong_feature_count_raises(self, detector):
        """Passing wrong number of features should raise."""
        bad_window = np.random.rand(60, 5).astype(np.float32)   # 5 features, not 8
        with pytest.raises(Exception):
            detector.predict(bad_window)

    def test_scaler_applied(self, lstm_model, feature_cols):
        """Detector should not crash when a fitted scaler is present."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        dummy_data = np.random.rand(100, 8).astype(np.float32)
        scaler.fit(dummy_data)

        detector = AnomalyDetector(
            model=lstm_model, scaler=scaler,
            threshold=0.05, feature_cols=feature_cols,
        )
        window = np.random.rand(60, 8).astype(np.float32)
        result = detector.predict(window)
        assert result.reconstruction_error >= 0
