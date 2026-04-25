# Architecture Document

## System Overview

The Multivariate Log Anomaly Detection System is a production-grade MLOps application that continuously monitors server health metrics and automatically flags anomalies using an LSTM Autoencoder model.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Data Layer                           │
│                                                             │
│   Node Exporter ──► Prometheus ──► Airflow DAG             │
│   (host metrics)    (scrape)       (every 5 min)           │
│                                         │                   │
│                                    SQLite DB                │
│                                    + daily CSV              │
│                                    (DVC tracked)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        ML Layer                             │
│                                                             │
│   preprocess.py ──► train.py ──► MLflow registry          │
│   (scale, window)   (LSTM AE)    (model versioning)        │
│                                         │                   │
│                        DVC (pipeline + data versioning)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Serving Layer                          │
│                                                             │
│   FastAPI backend (port 8000)                              │
│   ├── POST /predict   (inference)                          │
│   ├── GET  /health    (model status)                       │
│   ├── GET  /ready     (readiness probe)                    │
│   ├── POST /drift     (drift detection)                    │
│   ├── GET  /pipeline/status                                │
│   └── GET  /metrics   (Prometheus exporter)                │
└─────────────────────────────────────────────────────────────┘
                              │  REST API only
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Frontend Layer                          │
│                                                             │
│   Flask app (port 5000)                                    │
│   ├── /dashboard   (live metrics + anomaly table)          │
│   ├── /pipeline    (ML pipeline visualization)             │
│   └── /manual      (user manual)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                         │
│                                                             │
│   Prometheus (port 9090) ──► Grafana (port 3000)          │
│   scrapes /metrics every 15s   (NRT dashboards)           │
└─────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Node Exporter
Runs on the host machine. Exposes raw Linux system metrics (CPU, memory, disk, network) to Prometheus via HTTP.

### Prometheus
Time-series database and scraper. Pulls metrics from Node Exporter (every 15s) and from the FastAPI `/metrics` endpoint. Stores up to 15 days of metric history.

### Airflow DAG (`ingest_prometheus_metrics`)
Scheduled every 5 minutes. Three tasks: scrape Prometheus instant query → validate schema → write to SQLite and daily CSV. CSVs are DVC-tracked for reproducibility.

### DVC
Tracks the data pipeline (preprocess → train → evaluate) and raw/processed data files. Ensures any training run is reproducible from a Git commit hash + DVC lock file.

### MLflow
Tracks all training experiments (hyperparameters, loss curves, threshold). Stores trained models in a model registry. The backend always loads the `latest` registered version.

### FastAPI Backend
Stateless inference engine. Loads the MLflow-registered LSTM model at startup. Exposes clean REST endpoints. Instruments all operations with Prometheus counters and histograms. Persists predictions to SQLite.

### Flask Frontend
Thin presentation layer. Communicates with the backend exclusively via REST API. Three screens: dashboard, pipeline visualisation, user manual.

### Grafana
Reads from Prometheus. Pre-provisioned dashboard shows inference latency, anomaly counts by severity, and reconstruction error over time.

## Design Decisions

- **Loose coupling**: Frontend and backend are independent Docker services connected only via configurable REST API. No shared imports.
- **SQLite for simplicity**: No cloud services required. SQLite stores metrics and inference results locally, meeting the no-cloud constraint.
- **Threshold at p95**: The 95th percentile of training reconstruction errors is a conservative, data-driven threshold that minimises false positives on normal traffic.
- **Two severity levels**: Low (1x–2x threshold) and High (>2x threshold) keep the alert surface simple and actionable.
