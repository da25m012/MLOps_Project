"""
FastAPI backend — inference engine.
Endpoints:
  POST /predict          → run anomaly detection on a metric window
  GET  /health           → model load status
  GET  /ready            → orchestration readiness check
  POST /drift            → check for data drift vs baseline
  GET  /pipeline/status  → pipeline stats for Flask pipeline visualization screen
  GET  /metrics          → Prometheus scrape endpoint
"""

import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import Response

import model as model_loader
from exceptions import (
    InvalidWindowError,
    ModelNotReadyError,
    generic_exception_handler,
    invalid_window_handler,
    model_not_ready_handler,
)
from metrics import (
    active_requests,
    anomaly_detections_total,
    get_metrics,
    inference_latency_seconds,
    inference_requests_total,
    model_reconstruction_error,
)
from schemas import (
    AnomalyResult,
    DriftReport,
    DriftResponse,
    HealthResponse,
    MetricWindow,
    PipelineStatus,
    ReadyResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.environ.get("DB_PATH", "/app/data/metrics.db")
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", "/app/data/processed")
SEQ_LEN = 60
FEATURES = ["cpu_usage", "memory_usage", "request_rate", "error_rate"]


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model...")
    model_loader.load_model()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Anomaly Detection API",
    description="LSTM Autoencoder inference backend for multivariate log anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

# Register exception handlers
app.add_exception_handler(ModelNotReadyError, model_not_ready_handler)
app.add_exception_handler(InvalidWindowError, invalid_window_handler)
app.add_exception_handler(Exception, generic_exception_handler)


# ── Helper ────────────────────────────────────────────────────────────────────

def validate_window(window: list):
    if len(window) != SEQ_LEN:
        raise InvalidWindowError(f"Window must have {SEQ_LEN} rows, got {len(window)}.")
    for row in window:
        if len(row) != len(FEATURES):
            raise InvalidWindowError(f"Each row must have {len(FEATURES)} values, got {len(row)}.")


def get_db():
    return sqlite3.connect(DB_PATH)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=AnomalyResult, summary="Run anomaly detection")
async def predict(payload: MetricWindow):
    """
    Accepts a window of raw system metrics and returns anomaly classification.

    - **window**: 2D list of shape (60, 4) — [cpu, memory, req_rate, error_rate]
    - **timestamp**: optional ISO timestamp of the last data point
    """
    if not model_loader.is_ready():
        raise ModelNotReadyError("Model is not loaded. Check /health for details.")

    validate_window(payload.window)

    inference_requests_total.inc()
    active_requests.inc()
    start = time.time()

    try:
        result = model_loader.predict(payload.window)
    finally:
        elapsed = time.time() - start
        inference_latency_seconds.observe(elapsed)
        active_requests.dec()

    model_reconstruction_error.set(result["score"])
    anomaly_detections_total.labels(severity=result["severity"]).inc()

    # Persist result to SQLite
    try:
        with get_db() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO anomaly_results
                   (timestamp, is_anomaly, severity, score) VALUES (?, ?, ?, ?)""",
                (payload.timestamp or datetime.utcnow().isoformat(), result["is_anomaly"],
                 result["severity"], result["score"]),
            )
    except Exception as e:
        logger.warning(f"Failed to persist result to DB: {e}")

    return AnomalyResult(
        is_anomaly=result["is_anomaly"],
        severity=result["severity"],
        score=result["score"],
        threshold=result["threshold"],
        timestamp=payload.timestamp,
    )


@app.get("/health", response_model=HealthResponse, summary="Model health check")
async def health():
    """Returns model load status and MLflow run ID."""
    return HealthResponse(
        status="ok" if model_loader.is_ready() else "degraded",
        model_loaded=model_loader.is_ready(),
        mlflow_run_id=model_loader.get_run_id(),
    )


@app.get("/ready", response_model=ReadyResponse, summary="Readiness probe")
async def ready():
    """Used by Docker/orchestrators to check if the service is ready to accept traffic."""
    if model_loader.is_ready():
        return ReadyResponse(ready=True, detail="Model loaded and ready.")
    return ReadyResponse(ready=False, detail="Model not yet loaded.")


@app.post("/drift", response_model=DriftResponse, summary="Data drift detection")
async def drift(payload: MetricWindow):
    """
    Computes per-feature statistics on the provided window and
    compares them against the training baseline to detect drift.
    """
    validate_window(payload.window)
    arr = np.array(payload.window)
    live_stats = {
        feat: {"mean": float(arr[:, i].mean()), "std": float(arr[:, i].std())}
        for i, feat in enumerate(FEATURES)
    }

    from evaluate import detect_drift
    report = detect_drift(live_stats)

    drift_reports = [
        DriftReport(
            feature=feat,
            drift_detected=info["drift_detected"],
            z_score=info["z_score"],
            baseline_mean=info["baseline_mean"],
            live_mean=info["live_mean"],
        )
        for feat, info in report.items()
    ]

    return DriftResponse(
        drift_reports=drift_reports,
        any_drift=any(r.drift_detected for r in drift_reports),
    )


@app.get("/pipeline/status", response_model=PipelineStatus, summary="ML pipeline status")
async def pipeline_status():
    """
    Returns pipeline metadata for the Flask pipeline visualization screen.
    Reads from SQLite and the processed/ artifacts.
    """
    status = PipelineStatus()

    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT MAX(timestamp), COUNT(*) FROM metrics"
            ).fetchone()
            if row:
                status.last_ingestion = row[0]
                status.total_rows_ingested = row[1] or 0

            cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            anomaly_row = conn.execute(
                "SELECT COUNT(*) FROM anomaly_results WHERE is_anomaly=1 AND timestamp >= ?",
                (cutoff,)
            ).fetchone()
            total_row = conn.execute(
                "SELECT COUNT(*) FROM anomaly_results WHERE timestamp >= ?",
                (cutoff,)
            ).fetchone()
            if total_row and total_row[0]:
                status.anomaly_rate_24h = round((anomaly_row[0] or 0) / total_row[0], 4)
    except Exception as e:
        logger.warning(f"DB read failed for pipeline/status: {e}")

    threshold_path = os.path.join(PROCESSED_DIR, "threshold.txt")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            status.threshold = float(f.read().strip())

    if model_loader.get_run_id():
        status.model_version = model_loader.get_run_id()[:8]

    return status


@app.get("/metrics", summary="Prometheus metrics")
async def metrics():
    """Prometheus scrape endpoint — exposes instrumentation counters and gauges."""
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)
