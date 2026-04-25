"""
services/drift_service.py
==========================
Computes PSI-based data drift scores by comparing current
feature distributions against the saved reference baseline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

from app.core.config import settings
from app.models.schemas import DriftReport, FeatureDriftStats

log = logging.getLogger(__name__)

BASELINE_PATH = Path(settings.DATA_PATH) / "baselines"


class DriftService:

    def get_drift_report(self) -> DriftReport:
        reference = self._load_json(BASELINE_PATH / "reference_baseline.json")
        current = self._load_json(BASELINE_PATH / "current_baseline.json")

        feature_stats: List[FeatureDriftStats] = []
        features_with_drift: List[str] = []

        for feature in settings.FEATURE_COLS:
            ref = reference.get(feature, {})
            cur = current.get(feature, {})

            ref_mean = ref.get("mean", 0.0)
            cur_mean = cur.get("mean", 0.0)
            ref_std = ref.get("std", 1.0) or 1.0
            cur_std = cur.get("std", 1.0)

            psi = abs(cur_mean - ref_mean) / ref_std
            drift = psi > settings.DRIFT_PSI_THRESHOLD

            if drift:
                features_with_drift.append(feature)

            feature_stats.append(FeatureDriftStats(
                feature=feature,
                psi_score=round(psi, 4),
                drift_detected=drift,
                current_mean=round(cur_mean, 4),
                reference_mean=round(ref_mean, 4),
                current_std=round(cur_std, 4),
                reference_std=round(ref_std, 4),
            ))

        overall = len(features_with_drift) > 0
        recommendation = (
            "Model retraining recommended — drift detected on: " + ", ".join(features_with_drift)
            if overall else "No action required — all features within normal distribution"
        )

        return DriftReport(
            generated_at=datetime.utcnow(),
            overall_drift_detected=overall,
            features_with_drift=features_with_drift,
            feature_stats=feature_stats,
            psi_threshold=settings.DRIFT_PSI_THRESHOLD,
            recommendation=recommendation,
        )

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            log.warning("Baseline file not found: %s", path)
            return {}
        with open(path) as f:
            return json.load(f)
