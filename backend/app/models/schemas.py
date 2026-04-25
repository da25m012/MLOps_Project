"""
models/schemas.py
==================
Pydantic v2 schemas for all API request and response bodies.
These form the Low-Level Design (LLD) API contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ═════════════════════════════════════════════════════════════════════════════
# Health & readiness
# ═════════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str
    timestamp: datetime


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    model_server_reachable: bool
    mlflow_reachable: bool
    details: Dict[str, str] = Field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Prediction
# ═════════════════════════════════════════════════════════════════════════════

class MetricWindow(BaseModel):
    """
    A sliding window of system metric readings.
    Each inner list is one timestep with values in the order:
      [cpu_percent, mem_percent, disk_io_read_mb, disk_io_write_mb,
       net_bytes_sent_mb, net_bytes_recv_mb, req_rate_per_sec, error_rate_percent]
    """
    window: List[List[float]] = Field(
        ...,
        description="2-D array: (window_size, n_features)",
        min_length=1,
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="UTC timestamp of the window's last reading (defaults to now)",
    )

    @field_validator("window")
    @classmethod
    def validate_window(cls, v: List[List[float]]) -> List[List[float]]:
        if not v:
            raise ValueError("window must not be empty")
        n_features = len(v[0])
        if n_features != 8:
            raise ValueError(f"Each timestep must have 8 features, got {n_features}")
        for row in v:
            if len(row) != n_features:
                raise ValueError("All timesteps must have the same number of features")
        return v


class AnomalyPrediction(BaseModel):
    timestamp: datetime
    reconstruction_error: float = Field(..., description="Mean MSE over the window")
    threshold: float
    is_anomaly: bool
    severity: int = Field(..., ge=0, le=3, description="0=normal 1=low 2=medium 3=high")
    severity_label: str = Field(..., examples=["normal", "low", "medium", "high"])
    per_feature_errors: Dict[str, float] = Field(
        ..., description="Per-feature reconstruction errors"
    )


class BatchPredictRequest(BaseModel):
    windows: List[MetricWindow] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: List[AnomalyPrediction]
    total: int
    anomaly_count: int
    anomaly_rate_percent: float


# ═════════════════════════════════════════════════════════════════════════════
# Drift
# ═════════════════════════════════════════════════════════════════════════════

class FeatureDriftStats(BaseModel):
    feature: str
    psi_score: float
    drift_detected: bool
    current_mean: float
    reference_mean: float
    current_std: float
    reference_std: float


class DriftReport(BaseModel):
    generated_at: datetime
    overall_drift_detected: bool
    features_with_drift: List[str]
    feature_stats: List[FeatureDriftStats]
    psi_threshold: float
    recommendation: str


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline status
# ═════════════════════════════════════════════════════════════════════════════

class DagRunInfo(BaseModel):
    dag_id: str
    run_id: str
    state: str   # "success" | "running" | "failed" | "queued"
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    duration_seconds: Optional[float]


class ModelVersionInfo(BaseModel):
    name: str
    version: str
    stage: str
    run_id: str
    created_at: Optional[datetime]
    anomaly_threshold: Optional[float]
    val_loss: Optional[float]


class PipelineStatusResponse(BaseModel):
    timestamp: datetime
    ingestion_dag: Optional[DagRunInfo]
    training_dag: Optional[DagRunInfo]
    current_model: Optional[ModelVersionInfo]
    mlflow_experiment: str
    total_runs: int
    last_retrain_trigger: Optional[str]   # "drift" | "first_run" | "manual"


# ═════════════════════════════════════════════════════════════════════════════
# Metrics (time-series data for frontend charts)
# ═════════════════════════════════════════════════════════════════════════════

class MetricPoint(BaseModel):
    timestamp: datetime
    value: float


class MetricSeries(BaseModel):
    name: str
    unit: str
    points: List[MetricPoint]


class MetricsResponse(BaseModel):
    series: List[MetricSeries]
    from_ts: datetime
    to_ts: datetime
    resolution_seconds: int


# ═════════════════════════════════════════════════════════════════════════════
# Feedback loop (ground truth labels)
# ═════════════════════════════════════════════════════════════════════════════

class GroundTruthLabel(BaseModel):
    window_timestamp: datetime
    is_anomaly: bool
    notes: Optional[str] = None


class FeedbackResponse(BaseModel):
    accepted: bool
    message: str
