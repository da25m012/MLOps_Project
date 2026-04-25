"""
api/routes/metrics.py
======================
GET /api/v1/metrics  — time-series metric data for frontend charts
GET /api/v1/metrics/latest — latest single reading
GET /api/v1/metrics/anomalies — recent anomaly events only
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query

from app.models.schemas import MetricPoint, MetricSeries, MetricsResponse
from app.services.metrics_store import MetricEntry, metrics_store

log = logging.getLogger(__name__)
router = APIRouter()

# Feature names in the same order as MetricEntry fields
SERIES_META = [
    ("cpu_percent", "percent"),
    ("mem_percent", "percent"),
    ("disk_io_read_mb", "MB"),
    ("disk_io_write_mb", "MB"),
    ("net_bytes_sent_mb", "MB"),
    ("net_bytes_recv_mb", "MB"),
    ("req_rate_per_sec", "req/s"),
    ("error_rate_percent", "percent"),
    ("reconstruction_error", "MSE"),
]


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Time-series metric data for dashboard charts",
)
async def get_metrics(
    minutes: int = Query(default=30, ge=1, le=1440, description="Lookback window in minutes"),
    resolution: int = Query(default=10, ge=5, le=300, description="Resolution in seconds"),
):
    """
    Returns all metric series for the requested lookback window.
    Data is read from the in-memory rolling store populated by /predict.
    """
    now = datetime.utcnow()
    from_ts = now - timedelta(minutes=minutes)

    entries = metrics_store.query(from_ts=from_ts, to_ts=now)
    entries = _downsample(entries, resolution_seconds=resolution)

    series = []
    for field_name, unit in SERIES_META:
        points = [
            MetricPoint(timestamp=e.timestamp, value=getattr(e, field_name))
            for e in entries
        ]
        series.append(MetricSeries(name=field_name, unit=unit, points=points))

    return MetricsResponse(
        series=series,
        from_ts=from_ts,
        to_ts=now,
        resolution_seconds=resolution,
    )


@router.get("/metrics/latest", summary="Most recent single metric reading")
async def get_latest():
    entries = metrics_store.latest(1)
    if not entries:
        return {"message": "No data yet"}
    e = entries[0]
    return {
        "timestamp": e.timestamp,
        "cpu_percent": e.cpu_percent,
        "mem_percent": e.mem_percent,
        "error_rate_percent": e.error_rate_percent,
        "req_rate_per_sec": e.req_rate_per_sec,
        "reconstruction_error": e.reconstruction_error,
        "is_anomaly": e.is_anomaly,
        "severity": e.severity,
    }


@router.get("/metrics/anomalies", summary="Recent anomaly events")
async def get_anomalies(
    minutes: int = Query(default=60, ge=1, le=1440),
    severity_min: int = Query(default=1, ge=0, le=3),
):
    now = datetime.utcnow()
    from_ts = now - timedelta(minutes=minutes)
    entries = metrics_store.query(from_ts=from_ts, to_ts=now)
    anomalies = [
        {
            "timestamp": e.timestamp,
            "reconstruction_error": e.reconstruction_error,
            "severity": e.severity,
            "cpu_percent": e.cpu_percent,
            "error_rate_percent": e.error_rate_percent,
        }
        for e in entries
        if e.is_anomaly and e.severity >= severity_min
    ]
    return {"anomalies": anomalies, "count": len(anomalies)}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _downsample(
    entries: list[MetricEntry], resolution_seconds: int
) -> list[MetricEntry]:
    """Keep one entry per resolution bucket (last value wins)."""
    if not entries:
        return []
    buckets: dict[int, MetricEntry] = {}
    for e in entries:
        bucket = int(e.timestamp.timestamp()) // resolution_seconds
        buckets[bucket] = e
    return [buckets[k] for k in sorted(buckets)]
