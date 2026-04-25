"""
API I/O schemas.
All endpoint request and response bodies are defined here —
keeping them separate from business logic for clean LLD documentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Request schemas ───────────────────────────────────────────────────────────

class MetricWindow(BaseModel):
    """
    A sliding window of multivariate system metrics.
    Shape: seq_len rows × 4 features.
    Feature order: [cpu_usage, memory_usage, request_rate, error_rate]
    """
    window: List[List[float]] = Field(
        ...,
        description="2D list: (seq_len, 4). Values are raw (unscaled).",
        example=[[0.45, 0.60, 120.0, 0.01]] * 60,
    )
    timestamp: Optional[str] = Field(
        None,
        description="ISO 8601 timestamp of the last data point in the window.",
        example="2024-01-15T10:30:00Z",
    )


# ── Response schemas ──────────────────────────────────────────────────────────

class AnomalyResult(BaseModel):
    """Result of a single anomaly detection inference."""
    is_anomaly: bool = Field(..., description="True if the window is anomalous.")
    severity: str = Field(..., description="'normal', 'low', or 'high'.")
    score: float = Field(..., description="Reconstruction error (MSE). Higher = more anomalous.")
    threshold: float = Field(..., description="Threshold used for classification.")
    timestamp: Optional[str] = Field(None, description="Echoed from request.")


class HealthResponse(BaseModel):
    """Response for /health endpoint."""
    status: str = Field(..., example="ok")
    model_loaded: bool
    mlflow_run_id: Optional[str] = None


class ReadyResponse(BaseModel):
    """Response for /ready endpoint (orchestration health check)."""
    ready: bool
    detail: str


class DriftReport(BaseModel):
    """Per-feature drift detection result."""
    feature: str
    drift_detected: bool
    z_score: float
    baseline_mean: float
    live_mean: float


class DriftResponse(BaseModel):
    """Response for /drift endpoint."""
    drift_reports: List[DriftReport]
    any_drift: bool


class PipelineStatus(BaseModel):
    """Response for /pipeline/status — used by Flask pipeline visualization."""
    last_ingestion: Optional[str] = None
    total_rows_ingested: int = 0
    model_version: Optional[str] = None
    threshold: Optional[float] = None
    anomaly_rate_24h: Optional[float] = None
