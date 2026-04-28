#!/bin/bash
set -e

python - <<'PYEOF2'
import sqlite3, os
db = os.environ.get("DB_PATH", "/app/data/metrics.db")
os.makedirs(os.path.dirname(db), exist_ok=True)
conn = sqlite3.connect(db)
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
conn.close()
print("Database initialized.")
PYEOF2

export PYTHONPATH="/app/app:/app/ml:${PYTHONPATH}"

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
