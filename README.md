# NASA CMAPSS Jet Engine Anomaly Detection System

An end-to-end MLOps project that detects jet engine anomalies using an LSTM Autoencoder trained on NASA's CMAPSS FD001 turbofan engine dataset.

## Architecture

```
test_FD001.txt → Airflow DAG (batch ingestion) → SQLite (nasa_metrics)
                                                        ↓
                                          LSTM Autoencoder (MLflow)
                                                        ↓
                                               FastAPI (/predict)
                                                        ↓
                                               Flask Dashboard
                                                        ↓
                                     Prometheus + Grafana (monitoring)
```

## Dataset

NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) FD001 subset:
- 100 training engines, 100 test engines
- 21 sensors per cycle → 14 informative sensors selected
- Training: early cycles (cycle ≤ 125) = normal behaviour
- Anomaly: late cycles (near failure) = high reconstruction error

## Tech Stack

| Layer            | Tool                          |
|------------------|-------------------------------|
| Data ingestion   | Apache Airflow                |
| Versioning       | Git + DVC                     |
| ML framework     | PyTorch (LSTM Autoencoder)    |
| Experiment track | MLflow                        |
| Model serving    | FastAPI + Uvicorn             |
| Frontend         | Flask                         |
| Monitoring       | Prometheus + Grafana          |
| Containerisation | Docker + Docker Compose       |

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Git and DVC (`pip install dvc`)
- NASA CMAPSS FD001 files: `train_FD001.txt`, `test_FD001.txt` in `data/raw/`

## Quick Start

### 1. Clone and initialise

```bash
git clone <your-repo-url>
cd MLOps_project
git init
dvc init
```

Go to Kaggle NASA CMapss Jet Engine dataset and download train_FD001.txt and test_FD001.txt files and place them in data/raw/ folder.

### 2. Start core services

```bash
docker compose up -d
```

Services started:
- Flask frontend   → http://localhost:5000
- FastAPI backend  → http://localhost:8000
- Prometheus       → http://localhost:9090
- Grafana          → http://localhost:3000  (admin/admin)

### 3. Start MLflow locally

```bash
mlflow server --host 0.0.0.0 --port 5001 \
  --backend-store-uri sqlite:///mlflow_local.db \
  --default-artifact-root mlflow-artifacts:/ \
  --artifacts-destination ./mlruns \
  --serve-artifacts &
```

MLflow UI → http://localhost:5001

### 4. Start Airflow

```bash
cd airflow
docker compose -f docker-compose.airflow.yml up -d
```

Airflow UI → http://localhost:8080 (admin/admin)
Enable the `ingest_nasa_cmapss` DAG — it reads `test_FD001.txt` in batches every 5 minutes.

### 5. Train the model

```bash
cd ml
MLFLOW_TRACKING_URI=http://localhost:5001 python3 train.py
```

### 6. Restart backend to load trained model

```bash
cd ..
docker compose restart backend
curl http://localhost:8000/health
```

### 7. Run unit tests

```bash
cd backend
pip install pytest httpx
pytest tests/ -v
```

## API Endpoints

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| POST   | /predict           | Anomaly detection on a 30×14 sensor window |
| GET    | /health            | Model load status                        |
| GET    | /ready             | Orchestration readiness probe            |
| POST   | /drift             | Data drift detection vs training baseline |
| GET    | /pipeline/status   | Pipeline stats for UI                    |
| GET    | /metrics           | Prometheus scrape endpoint               |

## Input Format

```json
{
  "window": [[sensor2, sensor3, sensor4, sensor7, sensor8, sensor9,
               sensor11, sensor12, sensor13, sensor14, sensor15,
               sensor17, sensor20, sensor21], ...],
  "engine_id": 42,
  "cycle": 150
}
```

Window shape: (30, 14) — 30 engine cycles × 14 sensors.

## Anomaly Severity

| Severity | Condition                                  |
|----------|--------------------------------------------|
| normal   | Reconstruction error ≤ threshold           |
| low      | threshold < error ≤ 2 × threshold          |
| high     | error > 2 × threshold                      |

Threshold is set at the 95th percentile of reconstruction errors on normal training cycles.
