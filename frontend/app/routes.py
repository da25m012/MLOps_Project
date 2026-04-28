"""
Flask routes.
All backend communication goes through REST API calls to FastAPI.
Airflow REST API called directly for pipeline management console.
"""

import logging
import os
import base64

import requests
from flask import Blueprint, current_app, jsonify, render_template

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://localhost:8080")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.environ.get("AIRFLOW_PASS", "admin")
DAG_ID = "ingest_nasa_cmapss"


def _get_backend(path: str, timeout: int = 5):
    """Helper for backend GET requests."""
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        logger.error(f"Backend GET {path} failed: {e}")
        return None, str(e)


def _get_airflow(path: str, timeout: int = 5):
    """Helper for Airflow REST API GET requests."""
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


# ── Pages ─────────────────────────────────────────────────────────────────────

@bp.route("/")
@bp.route("/dashboard")
def dashboard():
    health, _ = _get_backend("/health")
    return render_template("dashboard.html", health=health)


@bp.route("/pipeline")
def pipeline():
    status, err = _get_backend("/pipeline/status")

    # Fetch DAG run history from Airflow
    dag_runs, dag_err = _get_airflow(f"/dags/{DAG_ID}/dagRuns?limit=20&order_by=-execution_date")
    task_instances = None

    # Fetch latest task instance statuses
    if dag_runs and dag_runs.get("dag_runs"):
        latest_run_id = dag_runs["dag_runs"][0].get("dag_run_id")
        if latest_run_id:
            task_instances, _ = _get_airflow(
                f"/dags/{DAG_ID}/dagRuns/{latest_run_id}/taskInstances"
            )

    return render_template(
        "pipeline.html",
        status=status,
        error=err,
        dag_runs=dag_runs.get("dag_runs", []) if dag_runs else [],
        dag_error=dag_err,
        task_instances=task_instances.get("task_instances", []) if task_instances else [],
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


@bp.route("/api/dag-runs")
def api_dag_runs():
    """Returns recent DAG run history for live refresh."""
    data, err = _get_airflow(f"/dags/{DAG_ID}/dagRuns?limit=20&order_by=-execution_date")
    if err:
        return jsonify({"error": err}), 502
    return jsonify(data.get("dag_runs", []))


@bp.route("/api/dag-runs/<run_id>/tasks")
def api_task_instances(run_id):
    """Returns task instances for a specific DAG run."""
    data, err = _get_airflow(f"/dags/{DAG_ID}/dagRuns/{run_id}/taskInstances")
    if err:
        return jsonify({"error": err}), 502
    return jsonify(data.get("task_instances", []))
