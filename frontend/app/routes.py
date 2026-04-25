"""
Flask routes.
All backend communication goes through REST API calls to FastAPI.
No direct model imports — strict loose coupling.
"""

import logging
import os

import requests
from flask import Blueprint, current_app, jsonify, render_template

logger = logging.getLogger(__name__)
bp = Blueprint("main", __name__)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://backend:8000")


def _get(path: str, timeout: int = 5):
    """Helper for backend GET requests."""
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        logger.error(f"Backend GET {path} failed: {e}")
        return None, str(e)


# ── Pages ─────────────────────────────────────────────────────────────────────

@bp.route("/")
@bp.route("/dashboard")
def dashboard():
    """Live metrics dashboard with anomaly table."""
    health, _ = _get("/health")
    return render_template("dashboard.html", health=health)


@bp.route("/pipeline")
def pipeline():
    """ML pipeline visualization screen."""
    status, err = _get("/pipeline/status")
    return render_template("pipeline.html", status=status, error=err)


@bp.route("/manual")
def manual():
    """User manual for non-technical users."""
    return render_template("manual.html")


# ── API proxy endpoints (called by dashboard JS via fetch) ────────────────────

@bp.route("/api/pipeline-status")
def api_pipeline_status():
    data, err = _get("/pipeline/status")
    if err:
        return jsonify({"error": err}), 502
    return jsonify(data)


@bp.route("/api/health")
def api_health():
    data, err = _get("/health")
    if err:
        return jsonify({"error": err}), 502
    return jsonify(data)
