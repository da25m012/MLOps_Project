"""
API I/O schemas for NASA CMAPSS anomaly detection.
14 sensor features per time step, 30 time steps per window.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

FEATURES = [
    "sensor2", "sensor3", "sensor4", "sensor7", "sensor8", "sensor9",
    "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
    "sensor17", "sensor20", "sensor21"
]
SEQ_LEN = 30
NUM_FEATURES = len(FEATURES)  # 14


class MetricWindow(BaseModel):
    """
    A sliding window of NASA CMAPSS sensor readings.
    Shape: seq_len=30 rows x 14 features.
    Feature order: sensor2,3,4,7,8,9,11,12,13,14,15,17,20,21
    """
    window: List[List[float]] = Field(
        ...,
        description=f"2D list: ({SEQ_LEN}, {NUM_FEATURES}). Raw unscaled sensor values.",
    )
    timestamp: Optional[str] = Field(
        None,
        description="ISO 8601 timestamp of the last data point.",
    )
    engine_id: Optional[int] = Field(
        None,
        description="Engine ID from the CMAPSS dataset.",
    )
    cycle: Optional[int] = Field(
        None,
        description="Current engine cycle number.",
    )


class AnomalyResult(BaseModel):
    is_anomaly: bool
    severity: str
    score: float
    threshold: float
    timestamp: Optional[str] = None
    engine_id: Optional[int] = None
    cycle: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mlflow_run_id: Optional[str] = None


class ReadyResponse(BaseModel):
    ready: bool
    detail: str


class DriftReport(BaseModel):
    feature: str
    drift_detected: bool
    z_score: float
    baseline_mean: float
    live_mean: float


class DriftResponse(BaseModel):
    drift_reports: List[DriftReport]
    any_drift: bool


class PipelineStatus(BaseModel):
    last_ingestion: Optional[str] = None
    total_rows_ingested: int = 0
    model_version: Optional[str] = None
    threshold: Optional[float] = None
    anomaly_rate_24h: Optional[float] = None
