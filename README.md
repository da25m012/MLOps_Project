# Multivariate Log Anomaly Detection System

An end-to-end MLOps project that detects infrastructure anomalies using an LSTM Autoencoder trained on Prometheus system metrics.

## Architecture

```
Prometheus → Airflow DAG → SQLite/CSV → DVC → LSTM Autoencoder (MLflow)
                                                       ↓
                                               FastAPI (/predict)
                                                       ↓
                                               Flask Dashboard
                                                       ↓
                                         Prometheus + Grafana (monitoring)
```

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

- Docker and Docker Compose installed
- Python 3.10+ (for running training locally)
- Git and DVC (`pip install dvc`)

## Quick Start

### 1. Clone and initialise

```bash
git clone <your-repo-url>
cd anomaly-detection
git init
dvc init
```

### 2. Start core services

```bash
docker compose up -d
```

This starts:
- Flask frontend   → http://localhost:5000
- FastAPI backend  → http://localhost:8000
- MLflow UI        → http://localhost:5001
- Prometheus       → http://localhost:9090
- Grafana          → http://localhost:3000  (admin / admin)

### 3. Start Airflow (separate compose)

```bash
cd airflow
docker compose -f docker-compose.airflow.yml up -d
```

Airflow UI → http://localhost:8080  
The `ingest_prometheus_metrics` DAG runs every 5 minutes automatically.

### 4. Train the model

Wait for at least a few hours of data collection (or seed with synthetic data), then:

```bash
# Run preprocessing + training via DVC
dvc repro

# Or run training directly
cd ml
pip install -r ../backend/requirements.txt
python train.py
```

Training logs metrics to MLflow. After training completes, the model is registered and the backend will load it automatically on next restart.

### 5. Verify everything works

```bash
# Backend health check
curl http://localhost:8000/health

# Run a test prediction (60 rows × 4 features)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"window": [[45.0, 60.0, 120.0, 0.02]] }'

# Check Prometheus metrics
curl http://localhost:8000/metrics
```

### 6. Run unit tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

## DVC Workflow

```bash
# Track new raw data files
dvc add data/raw/

# Run the full pipeline
dvc repro

# Show pipeline DAG
dvc dag

# Push data to remote (configure remote first)
dvc push
```

## Project Structure

```
anomaly-detection/
├── airflow/          # Airflow DAGs and config
├── backend/          # FastAPI inference engine
│   ├── app/          # main.py, model.py, schemas.py, metrics.py
│   └── tests/        # unit tests
├── frontend/         # Flask UI (dashboard, pipeline, manual)
├── ml/               # LSTM model, training, preprocessing
├── monitoring/       # Prometheus + Grafana config
├── data/             # DVC-tracked (gitignored)
├── docs/             # HLD, LLD, architecture, test plan, user manual
├── docker-compose.yml
└── dvc.yaml
```

## API Endpoints

| Method | Endpoint           | Description                        |
|--------|--------------------|------------------------------------|
| POST   | /predict           | Run anomaly detection on a window  |
| GET    | /health            | Model load status                  |
| GET    | /ready             | Orchestration readiness probe      |
| POST   | /drift             | Data drift detection               |
| GET    | /pipeline/status   | Pipeline stats for UI              |
| GET    | /metrics           | Prometheus scrape endpoint         |

## Anomaly Severity

| Severity | Condition                                  |
|----------|--------------------------------------------|
| normal   | Reconstruction error ≤ threshold           |
| low      | threshold < error ≤ 2 × threshold          |
| high     | error > 2 × threshold                      |

Threshold is set at the 95th percentile of reconstruction errors on training data.
