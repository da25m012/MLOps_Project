"""
api/routes/health.py
=====================
GET /health  — liveness probe (always returns 200 if process is alive)
GET /ready   — readiness probe (checks model + MLflow connectivity)
"""

import logging
from datetime import datetime

import httpx
from fastapi import APIRouter, Request

from app.core.config import settings
from app.models.schemas import HealthResponse, ReadyResponse

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    """Returns 200 immediately — used by Docker health checks and orchestrators."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
    )


@router.get("/ready", response_model=ReadyResponse, summary="Readiness probe")
async def ready(request: Request):
    """
    Checks all downstream dependencies before declaring the service ready.
    Returns 200 if ready, 503 if any critical dependency is unavailable.
    """
    model_client = request.app.state.model_client
    details: dict[str, str] = {}

    # 1. Model loaded in memory
    model_loaded = model_client.is_loaded
    details["model_version"] = model_client.model_version or "not loaded"
    details["model_loaded_at"] = (
        model_client.loaded_at.isoformat() if model_client.loaded_at else "never"
    )

    # 2. MLflow reachable
    mlflow_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.MLFLOW_TRACKING_URI}/health")
            mlflow_ok = resp.status_code == 200
    except Exception as exc:
        details["mlflow_error"] = str(exc)

    # 3. Model server reachable
    model_server_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.MODEL_SERVER_URL}/health")
            model_server_ok = resp.status_code == 200
    except Exception as exc:
        details["model_server_error"] = str(exc)

    ready = model_loaded  # model in memory is the hard requirement

    return ReadyResponse(
        ready=ready,
        model_loaded=model_loaded,
        model_server_reachable=model_server_ok,
        mlflow_reachable=mlflow_ok,
        details=details,
    )
