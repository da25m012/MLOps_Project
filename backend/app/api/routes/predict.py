"""
api/routes/predict.py
======================
POST /api/v1/predict       — single window prediction
POST /api/v1/predict/batch — batch prediction (up to 100 windows)
POST /api/v1/feedback      — submit ground truth label for feedback loop
"""

import asyncio
import logging
from datetime import datetime
from functools import partial

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import settings
from app.models.schemas import (
    AnomalyPrediction,
    BatchPredictRequest,
    BatchPredictResponse,
    FeedbackResponse,
    GroundTruthLabel,
    MetricWindow,
)
from app.services.metrics_store import MetricEntry, metrics_store
from scripts.metric_collector import MetricCollector

log = logging.getLogger(__name__)
router = APIRouter()

# Module-level Prometheus metric collector (updates gauges after each prediction)
_collector = MetricCollector()


def _get_model_client(request: Request):
    client = request.app.state.model_client
    if not client.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check /ready for details.",
        )
    return client


@router.post(
    "/predict",
    response_model=AnomalyPrediction,
    summary="Run anomaly detection on a single metric window",
)
async def predict(
    window: MetricWindow,
    request: Request,
):
    """
    Accepts a window of system metric readings and returns:
    - reconstruction error
    - anomaly flag + severity
    - per-feature error breakdown

    The window is also stored in the in-memory MetricsStore for
    the /metrics endpoint to serve to the frontend.
    """
    model_client = _get_model_client(request)

    try:
        # Run CPU-bound inference in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        prediction: AnomalyPrediction = await loop.run_in_executor(
            None, partial(model_client.predict, window)
        )
    except Exception as exc:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    # Update Prometheus gauges
    _collector.update_anomaly_metrics(
        reconstruction_error=prediction.reconstruction_error,
        is_anomaly=prediction.is_anomaly,
        severity=prediction.severity,
    )

    # Store in rolling metrics store
    raw = window.window[-1]  # use last row as "current" system state
    _store_entry(raw, prediction)

    log.info(
        "Prediction: error=%.6f anomaly=%s severity=%s",
        prediction.reconstruction_error,
        prediction.is_anomaly,
        prediction.severity_label,
    )
    return prediction


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Run anomaly detection on multiple windows",
)
async def predict_batch(
    body: BatchPredictRequest,
    request: Request,
):
    model_client = _get_model_client(request)
    predictions = []

    loop = asyncio.get_event_loop()
    for window in body.windows:
        pred = await loop.run_in_executor(
            None, partial(model_client.predict, window)
        )
        predictions.append(pred)

    anomaly_count = sum(1 for p in predictions if p.is_anomaly)
    anomaly_rate = (anomaly_count / len(predictions)) * 100 if predictions else 0.0

    return BatchPredictResponse(
        predictions=predictions,
        total=len(predictions),
        anomaly_count=anomaly_count,
        anomaly_rate_percent=round(anomaly_rate, 2),
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit ground truth label for feedback loop",
)
async def submit_feedback(label: GroundTruthLabel):
    """
    Logs a ground-truth anomaly label for a past window.
    Used by the feedback loop to accumulate labelled data for retraining.
    Labels are appended to data/feedback/ground_truth.jsonl.
    """
    import json
    from pathlib import Path

    feedback_path = Path(settings.DATA_PATH) / "feedback" / "ground_truth.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "window_timestamp": label.window_timestamp.isoformat(),
        "is_anomaly": label.is_anomaly,
        "notes": label.notes,
        "submitted_at": datetime.utcnow().isoformat(),
    }

    with open(feedback_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    log.info("Feedback recorded: %s", record)
    return FeedbackResponse(accepted=True, message="Ground truth label recorded successfully")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _store_entry(raw_row: list, prediction: AnomalyPrediction) -> None:
    """Append a MetricEntry to the rolling in-memory store."""
    try:
        entry = MetricEntry(
            timestamp=prediction.timestamp,
            cpu_percent=raw_row[0],
            mem_percent=raw_row[1],
            disk_io_read_mb=raw_row[2],
            disk_io_write_mb=raw_row[3],
            net_bytes_sent_mb=raw_row[4],
            net_bytes_recv_mb=raw_row[5],
            req_rate_per_sec=raw_row[6],
            error_rate_percent=raw_row[7],
            reconstruction_error=prediction.reconstruction_error,
            is_anomaly=prediction.is_anomaly,
            severity=prediction.severity,
        )
        metrics_store.append(entry)
    except Exception:
        log.warning("Failed to store metric entry", exc_info=True)
