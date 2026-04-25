"""api/routes/drift.py — GET /api/v1/drift-report"""

import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import DriftReport
from app.services.drift_service import DriftService

log = logging.getLogger(__name__)
router = APIRouter()
_service = DriftService()


@router.get(
    "/drift-report",
    response_model=DriftReport,
    summary="Per-feature data drift report (PSI-based)",
)
async def drift_report():
    """
    Compares current feature distributions against the reference baseline.
    Returns PSI scores per feature and an overall drift flag.
    Alerts are also emitted via Prometheus if PSI > threshold.
    """
    try:
        report = _service.get_drift_report()
        log.info(
            "Drift report: overall=%s features_drifted=%s",
            report.overall_drift_detected,
            report.features_with_drift,
        )
        return report
    except Exception as exc:
        log.exception("Failed to generate drift report")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
