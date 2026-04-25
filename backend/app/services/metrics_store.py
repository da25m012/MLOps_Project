"""
services/metrics_store.py
==========================
Thread-safe rolling in-memory store for recent metric readings.
The FastAPI backend appends to this store each time a prediction is made.
The /api/v1/metrics endpoint reads from here to serve the frontend charts.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, List, NamedTuple


class MetricEntry(NamedTuple):
    timestamp: datetime
    cpu_percent: float
    mem_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    net_bytes_sent_mb: float
    net_bytes_recv_mb: float
    req_rate_per_sec: float
    error_rate_percent: float
    reconstruction_error: float
    is_anomaly: bool
    severity: int


class MetricsStore:
    """Stores the last MAX_ENTRIES metric readings in a deque."""

    MAX_ENTRIES = 2000   # ~3 hours at 5-second intervals

    def __init__(self):
        self._lock = threading.Lock()
        self._data: Deque[MetricEntry] = deque(maxlen=self.MAX_ENTRIES)

    def append(self, entry: MetricEntry) -> None:
        with self._lock:
            self._data.append(entry)

    def query(
        self,
        from_ts: datetime,
        to_ts: datetime,
    ) -> List[MetricEntry]:
        with self._lock:
            return [e for e in self._data if from_ts <= e.timestamp <= to_ts]

    def latest(self, n: int = 1) -> List[MetricEntry]:
        with self._lock:
            data = list(self._data)
        return data[-n:] if data else []

    def all(self) -> List[MetricEntry]:
        with self._lock:
            return list(self._data)


# Module-level singleton
metrics_store = MetricsStore()
