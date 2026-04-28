"""
Flask routes.
All backend communication goes through REST API calls to FastAPI.
Airflow REST API called for pipeline management console (both DAGs).
"""

import logging
import os

import requests
from flask import Blueprint, jsonify, render_template
from .metrics import backend_health_status, track_request

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.environ.get("AIRFLOW_PASS", "admin")

INGESTION_DAG_ID = "ingest_nasa_cmapss"
TRAINING_DAG_ID  = "ml_training_pipeline"


def _get_backend(path: str, timeout: int = 5):
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Update health gauge based on /health response
        if path == "/health":
            backend_health_status.set(1 if data.get("status") == "ok" else 0)
        return data, None
    except Exception as e:
        logger.error(f"Backend GET {path} failed: {e}")
        backend_health_status.set(0)
        return None, str(e)


def _get_airflow(path: str, timeout: int = 5):
    try:
        resp = requests.get(
            f"{AIRFLOW_URL}/api/v1{path}",
            auth=(AIRFLOW_USER, AIRFLOW_PASS),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        logger.error(f"Airflow GET {path} failed: {e}")
        return None, str(e)


def _get_dag_runs(dag_id: str, limit: int = 10):
    data, err = _get_airflow(f"/dags/{dag_id}/dagRuns?limit={limit}&order_by=-execution_date")
    if data:
        return data.get("dag_runs", []), err
    return [], err


def _get_latest_task_instances(dag_id: str, dag_runs: list):
    if not dag_runs:
        return []
    latest_run_id = dag_runs[0].get("dag_run_id")
    if not latest_run_id:
        return []
    data, _ = _get_airflow(f"/dags/{dag_id}/dagRuns/{latest_run_id}/taskInstances")
    return data.get("task_instances", []) if data else []


# ── Pages ─────────────────────────────────────────────────────────────────────

@bp.route("/")
@bp.route("/dashboard")
@track_request
def dashboard():
    health, _ = _get_backend("/health")
    return render_template("dashboard.html", health=health)


@bp.route("/pipeline")
@track_request
def pipeline():
    status, err = _get_backend("/pipeline/status")

    # Ingestion DAG
    ingestion_runs, ingestion_err = _get_dag_runs(INGESTION_DAG_ID, limit=10)
    ingestion_tasks = _get_latest_task_instances(INGESTION_DAG_ID, ingestion_runs)

    # Training DAG
    training_runs, training_err = _get_dag_runs(TRAINING_DAG_ID, limit=10)
    training_tasks = _get_latest_task_instances(TRAINING_DAG_ID, training_runs)

    return render_template(
        "pipeline.html",
        status=status,
        error=err,
        ingestion_runs=ingestion_runs,
        ingestion_tasks=ingestion_tasks,
        ingestion_err=ingestion_err,
        training_runs=training_runs,
        training_tasks=training_tasks,
        training_err=training_err,
    )


@bp.route("/manual")
def manual():
    return render_template("manual.html")


# ── API proxy endpoints ────────────────────────────────────────────────────────

@bp.route("/api/pipeline-status")
def api_pipeline_status():
    data, err = _get_backend("/pipeline/status")
    if err:
        return jsonify({"error": err}), 502
    return jsonify(data)


@bp.route("/api/health")
def api_health():
    data, err = _get_backend("/health")
    if err:
        return jsonify({"error": err}), 502
    return jsonify(data)


@bp.route("/api/dag-runs/<dag_id>")
def api_dag_runs(dag_id):
    runs, err = _get_dag_runs(dag_id, limit=10)
    if err:
        return jsonify({"error": err}), 502
    return jsonify(runs)
