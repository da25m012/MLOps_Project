"""
Prometheus instrumentation for Flask frontend.
Tracks:
  - flask_requests_total{endpoint, method, status}  (counter)
  - flask_request_latency_seconds{endpoint}          (histogram)
  - backend_health_status                            (gauge: 1=ok, 0=degraded)
"""

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
import time
from functools import wraps
from flask import request, g

REGISTRY = CollectorRegistry(auto_describe=True)

flask_requests_total = Counter(
    "flask_requests_total",
    "Total Flask HTTP requests.",
    labelnames=["endpoint", "method", "status"],
    registry=REGISTRY,
)

flask_request_latency = Histogram(
    "flask_request_latency_seconds",
    "Flask request latency in seconds.",
    labelnames=["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0],
    registry=REGISTRY,
)

backend_health_status = Gauge(
    "backend_health_status",
    "Backend health: 1=ok, 0=degraded/unreachable.",
    registry=REGISTRY,
)


def get_metrics():
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def track_request(f):
    """Decorator to track request count and latency per endpoint."""
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        response = f(*args, **kwargs)
        elapsed = time.time() - start
        status = response.status_code if hasattr(response, 'status_code') else 200
        endpoint = request.endpoint or "unknown"
        flask_requests_total.labels(
            endpoint=endpoint, method=request.method, status=str(status)
        ).inc()
        flask_request_latency.labels(endpoint=endpoint).observe(elapsed)
        return response
    return decorated
