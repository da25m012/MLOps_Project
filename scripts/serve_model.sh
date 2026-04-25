#!/bin/bash
# serve_model.sh — starts the MLflow model server for the production LSTM model
set -e

MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://mlflow-server:5000}"
MODEL_NAME="${MODEL_NAME:-lstm-autoencoder}"
MODEL_STAGE="${MODEL_STAGE:-Production}"
PORT="${MODEL_SERVER_PORT:-5001}"

echo "[model-server] Waiting for MLflow at $MLFLOW_URI ..."
for i in $(seq 1 20); do
  if curl -sf "$MLFLOW_URI/health" > /dev/null 2>&1; then
    echo "[model-server] MLflow is ready."
    break
  fi
  echo "[model-server] Attempt $i/20 — retrying in 5s..."
  sleep 5
done

echo "[model-server] Starting MLflow model server on :$PORT"
exec mlflow models serve \
  --model-uri "models:/$MODEL_NAME/$MODEL_STAGE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --no-conda \
  --timeout 60
