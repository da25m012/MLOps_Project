"""
FastAPI backend — inference engine for NASA CMAPSS anomaly detection.
Endpoints:
  POST /predict          → anomaly detection on a 30×14 sensor window
  GET  /health           → model load status
  GET  /ready            → orchestration readiness probe
  POST /reload           → hot-swap model after retraining (no restart needed)
  POST /drift            → data drift detection vs training baseline
  GET  /pipeline/status  → pipeline stats for Flask pipeline visualization
  GET  /metrics          → Prometheus scrape endpoint
"""

import logging
import os
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import numpy as np
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import Response
import sqlite3
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML_DIR = "/app/ml"
for _p in [_HERE, _ML_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_loader
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
    FEATURES,
    SEQ_LEN,
    NUM_FEATURES,
)
from evaluate import detect_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "data"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(_DEFAULT_DATA_DIR, "metrics.db"))
PROCESSED_DIR = os.environ.get("PROCESSED_DIR", os.path.join(_DEFAULT_DATA_DIR, "processed"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading model...")
    model_loader.load_model()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="NASA CMAPSS Anomaly Detection API",
    description="LSTM Autoencoder inference backend for jet engine anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["*"],
)

app.add_exception_handler(ModelNotReadyError, model_not_ready_handler)
app.add_exception_handler(InvalidWindowError, invalid_window_handler)
app.add_exception_handler(Exception, generic_exception_handler)


def validate_window(window: list):
    if len(window) != SEQ_LEN:
        raise InvalidWindowError(f"Window must have {SEQ_LEN} rows, got {len(window)}.")
    for row in window:
        if len(row) != NUM_FEATURES:
            raise InvalidWindowError(f"Each row must have {NUM_FEATURES} values, got {len(row)}.")


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS nasa_metrics (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL UNIQUE,
            engine_id INTEGER,
            cycle     INTEGER,
            sensor2   REAL, sensor3  REAL, sensor4  REAL, sensor7  REAL,
            sensor8   REAL, sensor9  REAL, sensor11 REAL, sensor12 REAL,
            sensor13  REAL, sensor14 REAL, sensor15 REAL, sensor17 REAL,
            sensor20  REAL, sensor21 REAL
        );
        CREATE TABLE IF NOT EXISTS anomaly_results (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT NOT NULL UNIQUE,
            engine_id  INTEGER,
            cycle      INTEGER,
            is_anomaly INTEGER NOT NULL,
            severity   TEXT NOT NULL,
            score      REAL NOT NULL
        );
    """)
    return conn


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=AnomalyResult, summary="Run anomaly detection")
async def predict(payload: MetricWindow):
    """
    Accepts a 30×14 window of NASA CMAPSS sensor readings and returns anomaly classification.
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

    try:
        with get_db() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO anomaly_results
                   (timestamp, engine_id, cycle, is_anomaly, severity, score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (payload.timestamp or datetime.utcnow().isoformat(),
                 payload.engine_id, payload.cycle,
                 result["is_anomaly"], result["severity"], result["score"]),
            )
    except Exception as e:
        logger.warning(f"Failed to persist result to DB: {e}")

    return AnomalyResult(
        is_anomaly=result["is_anomaly"],
        severity=result["severity"],
        score=result["score"],
        threshold=result["threshold"],
        timestamp=payload.timestamp,
        engine_id=payload.engine_id,
        cycle=payload.cycle,
    )

@app.options("/predict")
async def options_predict(request: Request):
    return {}


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
    """Orchestration readiness probe — used by Docker health checks."""
    if model_loader.is_ready():
        return ReadyResponse(ready=True, detail="Model loaded and ready.")
    return ReadyResponse(ready=False, detail="Model not yet loaded.")


@app.post("/reload", summary="Hot-swap model after retraining")
async def reload():
    """
    Reloads the latest registered MLflow model into memory without restarting the server.
    Called automatically by the Airflow ml_training_pipeline DAG after training completes.
    """
    logger.info("Reload requested — loading latest model from MLflow...")
    try:
        model_loader.load_model()
        return {
            "status": "reloaded",
            "model_loaded": model_loader.is_ready(),
            "mlflow_run_id": model_loader.get_run_id(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Reload failed: {e}")
        return {"status": "failed", "error": str(e)}


@app.post("/drift", response_model=DriftResponse, summary="Data drift detection")
async def drift(payload: MetricWindow):
    """
    Computes per-feature stats on the window and compares against training baseline.
    """
    validate_window(payload.window)
    arr = np.array(payload.window)
    live_stats = {
        feat: {"mean": float(arr[:, i].mean()), "std": float(arr[:, i].std())}
        for i, feat in enumerate(FEATURES)
    }

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
    """Returns pipeline metadata for the Flask pipeline visualization screen."""
    status = PipelineStatus()
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT MAX(timestamp), COUNT(*) FROM nasa_metrics"
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
        logger.warning(f"DB read failed: {e}")

    threshold_path = os.path.join(PROCESSED_DIR, "threshold.txt")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            status.threshold = float(f.read().strip())

    if model_loader.get_run_id():
        status.model_version = model_loader.get_run_id()[:8]

    return status


@app.get("/metrics", summary="Prometheus metrics scrape endpoint")
async def metrics():
    """Exposes Prometheus instrumentation counters and gauges."""
    data, content_type = get_metrics()
    return Response(content=data, media_type=content_type)

import sqlite3
from datetime import datetime

@app.get("/latest-window")
def latest_window():
    conn = get_db()   # ✅ use your existing DB helper
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # latest engine
    cur.execute("""
        SELECT engine_id 
        FROM nasa_metrics 
        ORDER BY timestamp DESC 
        LIMIT 1
    """)
    row = cur.fetchone()
    if not row:
        return {"error": "no data"}

    engine_id = row["engine_id"]

    # last 30 rows
    cur.execute("""
        SELECT * FROM nasa_metrics
        WHERE engine_id = ?
        ORDER BY cycle DESC
        LIMIT 30
    """, (engine_id,))

    rows = cur.fetchall()
    rows = list(reversed(rows))

    if len(rows) < 30:
        return {"error": "not enough data"}

    # ✅ FIXED column names
    window = []
    for r in rows:
        window.append([
            r["sensor2"],
            r["sensor3"],
            r["sensor4"],
            r["sensor7"],
            r["sensor8"],
            r["sensor9"],
            r["sensor11"],
            r["sensor12"],
            r["sensor13"],
            r["sensor14"],
            r["sensor15"],
            r["sensor17"],
            r["sensor20"],
            r["sensor21"],
        ])

    return {
        "window": window,
        "engine_id": engine_id,
        "cycle": rows[-1]["cycle"],
        "timestamp": datetime.utcnow().isoformat()
    }