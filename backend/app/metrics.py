"""
Prometheus instrumentation.
Exposes /metrics endpoint for Prometheus to scrape.
Tracks:
  - inference_requests_total        (counter)
  - inference_latency_seconds       (histogram)
  - anomaly_detections_total        (counter, by severity)
  - model_reconstruction_error      (gauge, latest score)
  - active_requests                 (gauge)
"""

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry(auto_describe=True)

inference_requests_total = Counter(
    "inference_requests_total",
    "Total number of inference requests received.",
    registry=REGISTRY,
)

inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds.",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
    registry=REGISTRY,
)

anomaly_detections_total = Counter(
    "anomaly_detections_total",
    "Total anomalies detected, by severity.",
    labelnames=["severity"],
    registry=REGISTRY,
)

model_reconstruction_error = Gauge(
    "model_reconstruction_error",
    "Latest reconstruction error (anomaly score) from the model.",
    registry=REGISTRY,
)

active_requests = Gauge(
    "active_requests",
    "Number of inference requests currently being processed.",
    registry=REGISTRY,
)


def get_metrics() -> tuple[bytes, str]:
    """Returns Prometheus metrics payload and content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
