"""api/routes/pipeline.py — GET /api/v1/pipeline-status"""

import logging
from fastapi import APIRouter
from app.models.schemas import PipelineStatusResponse
from app.services.pipeline_service import PipelineService

log = logging.getLogger(__name__)
router = APIRouter()
_service = PipelineService()


@router.get(
    "/pipeline-status",
    response_model=PipelineStatusResponse,
    summary="Live Airflow DAG and MLflow model status",
)
async def pipeline_status():
    """
    Returns the latest run state for both Airflow DAGs and the
    currently deployed model version from the MLflow registry.
    Used by the ML Pipeline Visualization screen in the frontend.
    """
    return await _service.get_status()
