// services/api.js — centralised API client
const BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

async function get(path) {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`);
  return res.json();
}

async function post(path, body) {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`);
  return res.json();
}

// ── Health ──────────────────────────────────────────────────────────────────
export const fetchHealth = () => get('/health');
export const fetchReady  = () => get('/ready');

// ── Metrics history ─────────────────────────────────────────────────────────
export const fetchMetricsHistory = (minutes = 30, limit = 500) =>
  get(`/api/v1/metrics/history?minutes=${minutes}&limit=${limit}`);

export const fetchMetricsSummary = () => get('/api/v1/metrics/summary');
export const fetchMetricsLatest  = () => get('/api/v1/metrics/latest');

// ── Prediction ──────────────────────────────────────────────────────────────
export const predict = (window) => post('/api/v1/predict', { window });
export const predictBatch = (windows) => post('/api/v1/predict/batch', { windows });

// ── Drift ───────────────────────────────────────────────────────────────────
export const fetchDriftReport = () => get('/api/v1/drift-report');

// ── Pipeline status ─────────────────────────────────────────────────────────
export const fetchPipelineStatus = () => get('/api/v1/pipeline-status');
