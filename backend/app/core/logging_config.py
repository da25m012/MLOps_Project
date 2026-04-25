"""Structured JSON logging setup for production."""

import logging
import sys
from pythonjsonlogger import jsonlogger
from app.core.config import settings


def setup_logging() -> None:
    """Configure root logger with JSON formatter."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "mlflow"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
