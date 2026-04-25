"""
Airflow DAG: ingest_prometheus_metrics
Schedule: every 5 minutes
Tasks:
  1. scrape_prometheus  → fetch CPU, memory, req_rate, error_rate from Prometheus
  2. validate_data      → check for nulls and schema
  3. store_metrics      → write to SQLite DB and append to daily CSV (DVC-tracked)
"""

import csv
import logging
import os
import sqlite3
from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://prometheus:9090")
DB_PATH = os.environ.get("DB_PATH", "/app/data/metrics.db")
RAW_DIR = os.environ.get("RAW_DIR", "/app/data/raw")

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "start_date": datetime(2024, 1, 1),
}

# Prometheus queries for each metric
QUERIES = {
    "cpu_usage":    '100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
    "memory_usage": '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
    "request_rate": 'sum(rate(http_requests_total[5m]))',
    "error_rate":   'sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))',
}


def _query_prometheus(query: str) -> float:
    """Runs an instant Prometheus query and returns the scalar result."""
    try:
        resp = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()["data"]["result"]
        if not result:
            logger.warning(f"Empty result for query: {query}")
            return 0.0
        return float(result[0]["value"][1])
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
        return 0.0


def scrape_prometheus(**context):
    """Task 1: Scrape all metrics from Prometheus and push to XCom."""
    timestamp = datetime.utcnow().isoformat()
    metrics = {"timestamp": timestamp}
    for metric_name, query in QUERIES.items():
        metrics[metric_name] = _query_prometheus(query)
        logger.info(f"  {metric_name}: {metrics[metric_name]:.4f}")
    context["ti"].xcom_push(key="metrics", value=metrics)
    logger.info(f"Scraped metrics at {timestamp}")


def validate_data(**context):
    """Task 2: Validate scraped data — check for None values and plausible ranges."""
    metrics = context["ti"].xcom_pull(key="metrics", task_ids="scrape_prometheus")
    required_fields = ["cpu_usage", "memory_usage", "request_rate", "error_rate"]

    for field in required_fields:
        if metrics.get(field) is None:
            raise ValueError(f"Missing required field: {field}")

    # Sanity range checks
    if not (0 <= metrics["cpu_usage"] <= 100):
        logger.warning(f"cpu_usage out of range: {metrics['cpu_usage']}")
    if not (0 <= metrics["memory_usage"] <= 100):
        logger.warning(f"memory_usage out of range: {metrics['memory_usage']}")
    if metrics["error_rate"] < 0:
        logger.warning(f"error_rate negative: {metrics['error_rate']}")

    logger.info("Data validation passed.")


def store_metrics(**context):
    """Task 3: Persist validated metrics to SQLite and daily CSV."""
    metrics = context["ti"].xcom_pull(key="metrics", task_ids="scrape_prometheus")

    # ── SQLite ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp    TEXT NOT NULL UNIQUE,
                cpu_usage    REAL,
                memory_usage REAL,
                request_rate REAL,
                error_rate   REAL
            )
        """)
        conn.execute(
            "INSERT OR IGNORE INTO metrics (timestamp, cpu_usage, memory_usage, request_rate, error_rate) "
            "VALUES (?, ?, ?, ?, ?)",
            (metrics["timestamp"], metrics["cpu_usage"], metrics["memory_usage"],
             metrics["request_rate"], metrics["error_rate"]),
        )
        conn.commit()
        logger.info("Metrics stored in SQLite.")
    finally:
        conn.close()

    # ── Daily CSV (DVC-tracked) ───────────────────────────────────────────────
    os.makedirs(RAW_DIR, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    csv_path = os.path.join(RAW_DIR, f"metrics_{date_str}.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "cpu_usage", "memory_usage", "request_rate", "error_rate"],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

    logger.info(f"Metrics appended to {csv_path}")


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="ingest_prometheus_metrics",
    description="Scrape system metrics from Prometheus every 5 minutes",
    default_args=DEFAULT_ARGS,
    schedule_interval="*/5 * * * *",
    catchup=False,
    tags=["mlops", "ingestion"],
) as dag:

    t1 = PythonOperator(
        task_id="scrape_prometheus",
        python_callable=scrape_prometheus,
    )

    t2 = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    t3 = PythonOperator(
        task_id="store_metrics",
        python_callable=store_metrics,
    )

    t1 >> t2 >> t3
