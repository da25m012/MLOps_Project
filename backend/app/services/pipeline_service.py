"""
services/pipeline_service.py
=============================
Fetches live pipeline status from Airflow REST API and MLflow registry.
Used by /api/v1/pipeline-status endpoint.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import httpx

from app.models.schemas import DagRunInfo, ModelVersionInfo, PipelineStatusResponse

log = logging.getLogger(__name__)

AIRFLOW_BASE = os.getenv("AIRFLOW_BASE_URL", "http://airflow-webserver:8080")
AIRFLOW_USER = os.getenv("AIRFLOW_USER", "admin")
AIRFLOW_PASS = os.getenv("AIRFLOW_PASS", "admin")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")


class PipelineService:

    async def get_status(self) -> PipelineStatusResponse:
        async with httpx.AsyncClient(timeout=8.0) as client:
            ingestion = await self._get_latest_dag_run(client, "metric_ingestion_pipeline")
            training = await self._get_latest_dag_run(client, "model_training_pipeline")
            model_info = await self._get_model_info(client)
            total_runs = await self._get_run_count(client)

        return PipelineStatusResponse(
            timestamp=datetime.utcnow(),
            ingestion_dag=ingestion,
            training_dag=training,
            current_model=model_info,
            mlflow_experiment="anomaly-detection",
            total_runs=total_runs,
            last_retrain_trigger=training.run_id.split("_")[0] if training else None,
        )

    async def _get_latest_dag_run(
        self, client: httpx.AsyncClient, dag_id: str
    ) -> Optional[DagRunInfo]:
        try:
            resp = await client.get(
                f"{AIRFLOW_BASE}/api/v1/dags/{dag_id}/dagRuns",
                params={"limit": 1, "order_by": "-start_date"},
                auth=(AIRFLOW_USER, AIRFLOW_PASS),
            )
            resp.raise_for_status()
            runs = resp.json().get("dag_runs", [])
            if not runs:
                return None
            r = runs[0]
            start = datetime.fromisoformat(r["start_date"]) if r.get("start_date") else None
            end = datetime.fromisoformat(r["end_date"]) if r.get("end_date") else None
            duration = (end - start).total_seconds() if start and end else None
            return DagRunInfo(
                dag_id=dag_id,
                run_id=r["dag_run_id"],
                state=r["state"],
                start_date=start,
                end_date=end,
                duration_seconds=duration,
            )
        except Exception as exc:
            log.warning("Could not fetch DAG run for %s: %s", dag_id, exc)
            return None

    async def _get_model_info(self, client: httpx.AsyncClient) -> Optional[ModelVersionInfo]:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow_client = MlflowClient()
            versions = mlflow_client.get_latest_versions("lstm-autoencoder", stages=["Production"])
            if not versions:
                return None
            v = versions[0]
            run = mlflow_client.get_run(v.run_id)
            return ModelVersionInfo(
                name=v.name,
                version=v.version,
                stage=v.current_stage,
                run_id=v.run_id,
                created_at=datetime.fromtimestamp(v.creation_timestamp / 1000),
                anomaly_threshold=float(v.tags.get("anomaly_threshold", 0)),
                val_loss=float(run.data.metrics.get("val_loss", 0)),
            )
        except Exception as exc:
            log.warning("Could not fetch model info: %s", exc)
            return None

    async def _get_run_count(self, client: httpx.AsyncClient) -> int:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            from mlflow.tracking import MlflowClient
            mlflow_client = MlflowClient()
            exp = mlflow_client.get_experiment_by_name("anomaly-detection")
            if not exp:
                return 0
            runs = mlflow_client.search_runs(experiment_ids=[exp.experiment_id])
            return len(runs)
        except Exception:
            return 0
