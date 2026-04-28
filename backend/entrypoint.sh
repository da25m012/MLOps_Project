#!/bin/bash
set -e

python - <<'PYEOF'
import sqlite3, os
db = os.environ.get("DB_PATH", "/app/data/metrics.db")
os.makedirs(os.path.dirname(db), exist_ok=True)
conn = sqlite3.connect(db)
conn.executescript("""
CREATE TABLE IF NOT EXISTS metrics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL UNIQUE,
    cpu_usage     REAL,
    memory_usage  REAL,
    request_rate  REAL,
    error_rate    REAL
);
CREATE TABLE IF NOT EXISTS anomaly_results (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT NOT NULL,
    is_anomaly INTEGER NOT NULL,
    severity   TEXT NOT NULL,
    score      REAL NOT NULL
);
""")
conn.close()
print("Database initialized.")
PYEOF

# Make all app modules importable
export PYTHONPATH="/app/app:/app/ml:${PYTHONPATH}"

exec uvicorn app.main:app --host 0.0.0.0 --port 8000