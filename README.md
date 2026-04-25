# Multivariate Log Anomaly Detection System

Real-time anomaly detection for system metrics using an LSTM Autoencoder, built end-to-end with MLOps best practices.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Data Ingestion (Airflow)                               │
│  collect → validate → baselines → preprocess → DVC      │
└───────────────────────┬─────────────────────────────────┘
                        │ triggers (on drift / first run)
┌───────────────────────▼─────────────────────────────────┐
│  Model Training (Airflow + MLflow)                      │
│  load_features → train_lstm → evaluate → register        │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Serving (Docker Compose — 3 containers)                │
│  React frontend ↔ FastAPI backend ↔ MLflow model server  │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│  Monitoring (Prometheus + Grafana)                      │
│  Alerts: error rate > 5%, drift PSI > 0.2              │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Git + DVC (`pip install dvc`)

### 1. Clone and initialise
```bash
git clone <repo-url> && cd anomaly-detection
dvc init
cp .env.example .env
```

### 2. Start all services
```bash
docker compose up -d
```

### 3. Access the UIs

| Service | URL |
|---|---|
| React Dashboard | http://localhost:3000 |
| FastAPI docs | http://localhost:8000/docs |
| Airflow | http://localhost:8080 (admin/admin) |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3001 (admin/admin) |
| Prometheus | http://localhost:9090 |

### 4. Run the DVC pipeline (standalone, without Airflow)
```bash
dvc repro
```

### 5. Run tests
```bash
cd backend && pytest tests/ -v --tb=short
```

## Project Structure

```
anomaly-detection/
├── airflow/
│   ├── dags/
│   │   ├── metric_ingestion_pipeline.py   # main ingestion DAG
│   │   └── model_training_pipeline.py     # triggered training DAG
│   └── requirements.txt
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI route handlers
│   │   ├── core/         # config, logging
│   │   ├── models/       # Pydantic schemas, LSTM model class
│   │   ├── services/     # model client, drift service
│   │   └── main.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/   # Chart, AnomalyBadge, PipelineView, etc.
│       ├── pages/        # Dashboard, Pipeline, Settings
│       └── services/     # API client
├── monitoring/
│   ├── prometheus/       # prometheus.yml, alert_rules.yml
│   └── grafana/          # dashboards, provisioning
├── scripts/
│   └── metric_collector.py
├── data/                 # DVC-tracked (not in Git)
├── models/               # DVC-tracked (not in Git)
├── dvc.yaml              # DVC pipeline DAG
├── params.yaml           # All hyperparameters
└── docker-compose.yml
```

## Technology Stack

| Layer | Tool |
|---|---|
| Data ingestion | Apache Airflow 2.8 |
| Data versioning | DVC + Git LFS |
| Experiment tracking | MLflow 2.11 |
| Model | LSTM Autoencoder (PyTorch 2.2) |
| Model serving | MLflow model server |
| API | FastAPI + Uvicorn |
| Frontend | React 18 |
| Monitoring | Prometheus + Grafana |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions + DVC |

## MLOps Pipeline (DVC DAG)

```
collect → validate → preprocess ─┐
                 └→ compute_baselines
                                  └→ train → evaluate
```

View with: `dvc dag`

## Key Configuration

All tunable parameters live in `params.yaml` and are tracked by DVC and logged to MLflow on every run.

## Anomaly Severity Levels

| Level | Condition |
|---|---|
| Normal | recon_error ≤ threshold |
| Low | threshold < error ≤ threshold × 1.5 |
| Medium | threshold × 1.5 < error ≤ threshold × 2.5 |
| High | error > threshold × 2.5 |
