"""Application configuration — all settings from environment variables."""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── App ──────────────────────────────────────────────────────────────
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://frontend:80"]

    # ── Model server ─────────────────────────────────────────────────────
    MODEL_SERVER_URL: str = "http://model-server:5001"
    MODEL_NAME: str = "lstm-autoencoder"
    MODEL_STAGE: str = "Production"

    # ── MLflow ───────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://mlflow-server:5000"
    MLFLOW_EXPERIMENT_NAME: str = "anomaly-detection"

    # ── Anomaly thresholds ───────────────────────────────────────────────
    ANOMALY_THRESHOLD_PERCENTILE: float = 95.0   # trained threshold percentile
    SEVERITY_LOW_MULTIPLIER: float = 1.5          # recon_error > threshold * 1.5
    SEVERITY_MEDIUM_MULTIPLIER: float = 2.5
    SEVERITY_HIGH_MULTIPLIER: float = 4.0

    # ── Data ─────────────────────────────────────────────────────────────
    DATA_PATH: str = "/app/data"
    WINDOW_SIZE: int = 60          # seconds of metrics per window
    FEATURE_COLS: List[str] = [
        "cpu_percent", "mem_percent", "disk_io_read_mb",
        "disk_io_write_mb", "net_bytes_sent_mb", "net_bytes_recv_mb",
        "req_rate_per_sec", "error_rate_percent",
    ]

    # ── Drift detection ──────────────────────────────────────────────────
    DRIFT_PSI_THRESHOLD: float = 0.2   # Population Stability Index
    DRIFT_KS_ALPHA: float = 0.05       # KS test significance level


settings = Settings()
