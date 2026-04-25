"""
Multivariate Log Anomaly Detection System
FastAPI Application Entry Point
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import health, predict, pipeline, drift, metrics
from app.core.config import settings
from app.core.logging_config import setup_logging
from app.services.model_client import ModelClient

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    setup_logging()
    logger.info("Starting Anomaly Detection API", extra={"version": settings.APP_VERSION})

    # Initialise model client (validates model server connectivity)
    app.state.model_client = ModelClient(base_url=settings.MODEL_SERVER_URL)
    await app.state.model_client.check_health()
    logger.info("Model server connection established")

    yield

    logger.info("Shutting down Anomaly Detection API")
    await app.state.model_client.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Anomaly Detection API",
        description="Real-time multivariate log anomaly detection using LSTM Autoencoder",
        version=settings.APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Prometheus instrumentation ────────────────────────────────────────
    Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="anomaly_api_inprogress_requests",
        inprogress_labels=True,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(health.router, tags=["Health"])
    app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
    app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
    app.include_router(drift.router, prefix="/api/v1", tags=["Drift"])
    app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])

    return app


app = create_app()
