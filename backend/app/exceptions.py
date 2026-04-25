"""
Custom exception handlers.
Ensures all errors return consistent JSON responses with appropriate HTTP status codes.
"""

import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ModelNotReadyError(Exception):
    """Raised when inference is attempted but the model is not loaded."""
    pass


class InvalidWindowError(Exception):
    """Raised when the input window has wrong shape or values."""
    pass


async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    logger.error(f"ModelNotReadyError on {request.url}: {exc}")
    return JSONResponse(
        status_code=503,
        content={"error": "model_not_ready", "detail": str(exc)},
    )


async def invalid_window_handler(request: Request, exc: InvalidWindowError):
    logger.warning(f"InvalidWindowError on {request.url}: {exc}")
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_window", "detail": str(exc)},
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": "An unexpected error occurred."},
    )
