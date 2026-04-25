// pages/SettingsPage.js
import React from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchReady, fetchHealth } from '../services/api';

function ConfigRow({ label, value, note }) {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
      <div style={{ width: '44%', fontSize: '0.75rem', color: 'var(--text-secondary)', paddingTop: 1 }}>{label}</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--cyan)' }}>{value}</div>
        {note && <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>{note}</div>}
      </div>
    </div>
  );
}

function StatusRow({ label, ok, detail }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '10px 0', borderBottom: '1px solid var(--border)' }}>
      <span style={{ color: ok ? 'var(--normal)' : 'var(--high)', fontSize: '0.8rem' }}>{ok ? '●' : '○'}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-primary)' }}>{label}</div>
        {detail && <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{detail}</div>}
      </div>
      <span style={{ fontSize: '0.65rem', color: ok ? 'var(--normal)' : 'var(--high)', fontWeight: 700, letterSpacing: '0.1em' }}>
        {ok ? 'ONLINE' : 'OFFLINE'}
      </span>
    </div>
  );
}

export default function SettingsPage() {
  const { data: ready } = usePolling(fetchReady, 15000);
  const { data: health } = usePolling(fetchHealth, 15000);

  return (
    <div className="page-enter">
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ marginBottom: 4 }}>Settings & Status</h1>
        <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
          SYSTEM CONFIGURATION · SERVICE HEALTH · DOCUMENTATION LINKS
        </div>
      </div>

      <div className="grid-2" style={{ marginBottom: 20 }}>
        {/* ── Service health ───────────────────────────────────────── */}
        <div className="card">
          <div className="section-title" style={{ marginBottom: 12 }}>Service Health</div>
          <StatusRow label="FastAPI Backend"       ok={!!health}                         detail="api/health" />
          <StatusRow label="LSTM Model Server"     ok={ready?.model_server_reachable}    detail="model-server:5001" />
          <StatusRow label="Model Loaded"          ok={ready?.model_loaded}              detail={ready?.details?.model_version} />
          <StatusRow label="MLflow Tracking"       ok={ready?.mlflow_reachable}          detail="mlflow-server:5000" />
          <div style={{ marginTop: 14, fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            API version: {health?.version ?? '—'} · Last checked: {health?.timestamp ? new Date(health.timestamp).toLocaleTimeString() : '—'}
          </div>
        </div>

        {/* ── Model config ─────────────────────────────────────────── */}
        <div className="card">
          <div className="section-title" style={{ marginBottom: 12 }}>Model Configuration</div>
          <ConfigRow label="Architecture"     value="LSTM Autoencoder" note="Encoder → Latent → Decoder" />
          <ConfigRow label="Input Features"   value="8" note="cpu, mem, disk_r, disk_w, net_s, net_r, req_rate, err_rate" />
          <ConfigRow label="Window Size"      value="60 timesteps" note="Each window ≈ 5 minutes of data" />
          <ConfigRow label="Hidden Dim"       value="64" />
          <ConfigRow label="Latent Dim"       value="16" />
          <ConfigRow label="LSTM Layers"      value="2" />
          <ConfigRow label="Threshold"        value="p95 of training reconstruction errors" />
        </div>
      </div>

      {/* ── MLOps config ─────────────────────────────────────────────── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-title" style={{ marginBottom: 12 }}>MLOps Stack</div>
        <div className="grid-2">
          <div>
            <ConfigRow label="Data Ingestion"    value="Apache Airflow 2.8" note="DAG: metric_ingestion_pipeline (*/5 * * * *)" />
            <ConfigRow label="Data Versioning"   value="DVC + Git LFS" note="Tracks raw, processed, features, models" />
            <ConfigRow label="Experiment Track." value="MLflow 2.11" note="Params, metrics, artifacts per run" />
            <ConfigRow label="Model Registry"    value="MLflow Model Registry" note="Staging → Production with rollback" />
          </div>
          <div>
            <ConfigRow label="Containerisation"  value="Docker Compose" note="3 services + prometheus + grafana" />
            <ConfigRow label="Serving"           value="MLflow serve + FastAPI" note="REST endpoints on :8000" />
            <ConfigRow label="Monitoring"        value="Prometheus + Grafana" note="Alert: error rate > 5%, PSI > 0.2" />
            <ConfigRow label="CI/CD"             value="GitHub Actions + DVC" note=".github/workflows/ci.yml" />
          </div>
        </div>
      </div>

      {/* ── Severity legend ──────────────────────────────────────────── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-title" style={{ marginBottom: 12 }}>Anomaly Severity Levels</div>
        {[
          { level: 'Normal', color: 'var(--normal)', condition: 'recon_error ≤ threshold', action: 'No action required' },
          { level: 'Low',    color: 'var(--low)',    condition: 'threshold < error ≤ threshold × 1.5', action: 'Monitor closely' },
          { level: 'Medium', color: 'var(--medium)', condition: 'threshold × 1.5 < error ≤ threshold × 2.5', action: 'Investigate soon' },
          { level: 'High',   color: 'var(--high)',   condition: 'error > threshold × 2.5', action: 'Immediate action required' },
        ].map(row => (
          <div key={row.level} style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
            <span style={{ color: row.color, fontFamily: 'var(--font-display)', fontWeight: 700, width: 70 }}>{row.level}</span>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', flex: 1, fontFamily: 'var(--font-mono)' }}>{row.condition}</span>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{row.action}</span>
          </div>
        ))}
      </div>

      {/* ── Documentation links ──────────────────────────────────────── */}
      <div className="card">
        <div className="section-title" style={{ marginBottom: 12 }}>Documentation</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {[
            { label: 'API Reference (Swagger)',     url: 'http://localhost:8000/docs' },
            { label: 'API Reference (ReDoc)',        url: 'http://localhost:8000/redoc' },
            { label: 'Airflow DAG Console',          url: 'http://localhost:8080' },
            { label: 'MLflow Experiment Tracker',    url: 'http://localhost:5000' },
            { label: 'Grafana Dashboards',           url: 'http://localhost:3001' },
            { label: 'Prometheus Metrics',           url: 'http://localhost:9090' },
          ].map(link => (
            <a
              key={link.label}
              href={link.url}
              target="_blank"
              rel="noreferrer"
              style={{ display: 'block', padding: '8px 12px', background: 'var(--bg-elevated)', borderRadius: 'var(--r-sm)', color: 'var(--cyan)', fontSize: '0.72rem', textDecoration: 'none', fontFamily: 'var(--font-mono)', transition: 'background 0.15s' }}
              onMouseEnter={e => e.target.style.background = 'var(--bg-hover)'}
              onMouseLeave={e => e.target.style.background = 'var(--bg-elevated)'}
            >
              ↗ {link.label}
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
