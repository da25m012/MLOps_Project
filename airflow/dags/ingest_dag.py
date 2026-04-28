"""
Airflow DAG: ingest_nasa_cmapss
Reads NASA CMAPSS FD001 test data in batches and stores to SQLite.
Simulates streaming ingestion for the anomaly detection pipeline.
Schedule: every 5 minutes (reads next batch of engine cycles)
"""

import logging
import os
import sqlite3
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

RAW_DIR = os.environ.get("RAW_DIR", "/app/data/raw")
DB_PATH = os.environ.get("DB_PATH", "/app/data/metrics.db")

# 14 informative sensors
FEATURES = [
    "sensor2", "sensor3", "sensor4", "sensor7", "sensor8", "sensor9",
    "sensor11", "sensor12", "sensor13", "sensor14", "sensor15",
    "sensor17", "sensor20", "sensor21"
]

COLUMN_NAMES = (
    ["engine_id", "cycle", "op1", "op2", "op3"] +
    [f"sensor{i}" for i in range(1, 22)]
)

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "start_date": datetime(2024, 1, 1),
}


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nasa_metrics (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL UNIQUE,
            engine_id INTEGER,
            cycle     INTEGER,
            sensor2   REAL, sensor3  REAL, sensor4  REAL, sensor7  REAL,
            sensor8   REAL, sensor9  REAL, sensor11 REAL, sensor12 REAL,
            sensor13  REAL, sensor14 REAL, sensor15 REAL, sensor17 REAL,
            sensor20  REAL, sensor21 REAL
        )
    """)
    conn.commit()
    return conn


def load_batch(**context):
    """
    Reads next 10 rows from test_FD001.txt based on current offset.
    Stores offset in XCom for stateless operation.
    """
    import pandas as pd

    test_path = os.path.join(RAW_DIR, "test_FD001.txt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test_FD001.txt not found at {test_path}")

    df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)

    # Get current offset from DB row count
    conn = _get_db()
    row = conn.execute("SELECT COUNT(*) FROM nasa_metrics").fetchone()
    conn.close()
    offset = row[0] if row else 0

    batch = df.iloc[offset: offset + 10]
    if batch.empty:
        logger.info("All test data ingested. Restarting from beginning.")
        batch = df.iloc[0:10]

    records = batch[["engine_id", "cycle"] + FEATURES].to_dict(orient="records")
    context["ti"].xcom_push(key="batch", value=records)
    logger.info(f"Loaded batch of {len(records)} rows at offset {offset}")


def validate_batch(**context):
    """Validates that batch has no nulls and sensor values are finite."""
    import math
    records = context["ti"].xcom_pull(key="batch", task_ids="load_batch")
    for rec in records:
        for feat in FEATURES:
            val = rec.get(feat)
            if val is None or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Invalid value for {feat}: {val}")
    logger.info(f"Validated {len(records)} records successfully.")


def store_batch(**context):
    """Stores validated batch into SQLite."""
    records = context["ti"].xcom_pull(key="batch", task_ids="load_batch")
    conn = _get_db()
    inserted = 0
    for rec in records:
        timestamp = datetime.utcnow().isoformat() + f"_{rec['engine_id']}_{rec['cycle']}"
        try:
            conn.execute(
                """INSERT OR IGNORE INTO nasa_metrics
                   (timestamp, engine_id, cycle,
                    sensor2, sensor3, sensor4, sensor7, sensor8, sensor9,
                    sensor11, sensor12, sensor13, sensor14, sensor15,
                    sensor17, sensor20, sensor21)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (timestamp, rec["engine_id"], rec["cycle"],
                 rec["sensor2"], rec["sensor3"], rec["sensor4"],
                 rec["sensor7"], rec["sensor8"], rec["sensor9"],
                 rec["sensor11"], rec["sensor12"], rec["sensor13"],
                 rec["sensor14"], rec["sensor15"], rec["sensor17"],
                 rec["sensor20"], rec["sensor21"])
            )
            inserted += 1
        except Exception as e:
            logger.warning(f"Failed to insert record: {e}")
    conn.commit()
    conn.close()
    logger.info(f"Stored {inserted} records to SQLite.")


with DAG(
    dag_id="ingest_nasa_cmapss",
    description="Ingest NASA CMAPSS FD001 test data every 5 minutes",
    default_args=DEFAULT_ARGS,
    schedule_interval="*/5 * * * *",
    catchup=False,
    tags=["mlops", "ingestion", "nasa"],
) as dag:

    t1 = PythonOperator(task_id="load_batch", python_callable=load_batch)
    t2 = PythonOperator(task_id="validate_batch", python_callable=validate_batch)
    t3 = PythonOperator(task_id="store_batch", python_callable=store_batch)

    t1 >> t2 >> t3
