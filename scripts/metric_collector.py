"""
metric_collector.py
====================
Standalone Prometheus-compatible metric exporter.
Can be run as a long-lived process (exposes /metrics on :8001)
OR imported as a module for direct metric reading.

Metrics exported:
  - system_cpu_percent
  - system_mem_percent
  - system_disk_io_read_bytes_total
  - system_disk_io_write_bytes_total
  - system_net_bytes_sent_total
  - system_net_bytes_recv_total
  - app_request_rate_per_second   (simulated / replace with real app metrics)
  - app_error_rate_percent        (simulated / replace with real app metrics)
"""

import logging
import os
import time
from typing import Dict

import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    start_http_server,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

log = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────────────
registry = CollectorRegistry()

# System metrics
g_cpu = Gauge("system_cpu_percent", "CPU usage percent", registry=registry)
g_mem = Gauge("system_mem_percent", "Memory usage percent", registry=registry)
c_disk_read = Counter("system_disk_io_read_bytes_total", "Disk read bytes", registry=registry)
c_disk_write = Counter("system_disk_io_write_bytes_total", "Disk write bytes", registry=registry)
c_net_sent = Counter("system_net_bytes_sent_total", "Network bytes sent", registry=registry)
c_net_recv = Counter("system_net_bytes_recv_total", "Network bytes received", registry=registry)

# Application metrics (replace with real Prometheus scrape targets in prod)
g_req_rate = Gauge("app_request_rate_per_second", "HTTP request rate", registry=registry)
g_err_rate = Gauge("app_error_rate_percent", "HTTP error rate percent", registry=registry)

# Anomaly metrics (written by the backend when predictions are made)
g_anomaly_score = Gauge("model_anomaly_reconstruction_error",
                        "Latest reconstruction error from LSTM", registry=registry)
g_is_anomaly = Gauge("model_is_anomaly", "1 if current window is anomalous", registry=registry)
g_severity = Gauge("model_anomaly_severity",
                   "Severity level: 0=normal 1=low 2=medium 3=high", registry=registry)

# Drift metrics
g_psi = Gauge("data_drift_psi_score", "Population Stability Index per feature",
              ["feature"], registry=registry)


class MetricCollector:
    """Collects system metrics and updates Prometheus gauges."""

    def __init__(self):
        self._prev_disk = psutil.disk_io_counters()
        self._prev_net = psutil.net_io_counters()
        self._prev_ts = time.monotonic()

    def collect(self) -> Dict[str, float]:
        """Scrape current metrics and update gauges. Returns dict of values."""
        import numpy as np

        now = time.monotonic()
        elapsed = max(now - self._prev_ts, 0.001)
        self._prev_ts = now

        # CPU & memory
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        g_cpu.set(cpu)
        g_mem.set(mem)

        # Disk I/O (delta since last call)
        disk = psutil.disk_io_counters()
        if disk and self._prev_disk:
            disk_read_delta = max(disk.read_bytes - self._prev_disk.read_bytes, 0)
            disk_write_delta = max(disk.write_bytes - self._prev_disk.write_bytes, 0)
            c_disk_read.inc(disk_read_delta)
            c_disk_write.inc(disk_write_delta)
        self._prev_disk = disk

        # Network I/O (delta)
        net = psutil.net_io_counters()
        if net and self._prev_net:
            sent_delta = max(net.bytes_sent - self._prev_net.bytes_sent, 0)
            recv_delta = max(net.bytes_recv - self._prev_net.bytes_recv, 0)
            c_net_sent.inc(sent_delta)
            c_net_recv.inc(recv_delta)
        self._prev_net = net

        # Simulated app metrics (Poisson request rate, exponential error rate)
        req_rate = float(np.random.poisson(lam=120))
        err_rate = float(np.clip(np.random.exponential(scale=0.5), 0, 100))
        g_req_rate.set(req_rate)
        g_err_rate.set(err_rate)

        return {
            "cpu_percent": cpu,
            "mem_percent": mem,
            "disk_io_read_mb": disk_read_delta / 1e6 if disk else 0.0,
            "disk_io_write_mb": disk_write_delta / 1e6 if disk else 0.0,
            "net_bytes_sent_mb": sent_delta / 1e6 if net else 0.0,
            "net_bytes_recv_mb": recv_delta / 1e6 if net else 0.0,
            "req_rate_per_sec": req_rate,
            "error_rate_percent": err_rate,
        }

    def update_anomaly_metrics(
        self,
        reconstruction_error: float,
        is_anomaly: bool,
        severity: int,
    ) -> None:
        """Called by the FastAPI backend after each prediction."""
        g_anomaly_score.set(reconstruction_error)
        g_is_anomaly.set(int(is_anomaly))
        g_severity.set(severity)

    def update_drift_metrics(self, psi_scores: Dict[str, float]) -> None:
        """Update per-feature PSI gauges."""
        for feature, psi in psi_scores.items():
            g_psi.labels(feature=feature).set(psi)

    def get_metrics_text(self) -> bytes:
        """Return Prometheus text format for /metrics endpoint."""
        return generate_latest(registry)


# ── Standalone server mode ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("METRICS_PORT", "8001"))
    scrape_interval = float(os.getenv("SCRAPE_INTERVAL_SEC", "5"))

    log.info("Starting metric exporter on :%d (scrape every %.1fs)", port, scrape_interval)
    start_http_server(port, registry=registry)

    collector = MetricCollector()
    while True:
        metrics = collector.collect()
        log.debug("Collected: %s", metrics)
        time.sleep(scrape_interval)
