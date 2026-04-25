// pages/PipelinePage.js
import React from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchPipelineStatus } from '../services/api';
import PipelineViz from '../components/PipelineViz';
import StatCard from '../components/StatCard';

function ModelCard({ model }) {
  if (!model) {
    return (
      <div className="card" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: 32 }}>
        No production model found in registry
      </div>
    );
  }
  return (
    <div className="card">
      <div className="section-header">
        <span className="section-title">Production Model</span>
        <span className="badge badge-normal">● Live</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginTop: 4 }}>
        <div>
          <div className="card-label">Model Name</div>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--cyan)' }}>{model.name}</div>
        </div>
        <div>
          <div className="card-label">Version</div>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700 }}>v{model.version}</div>
        </div>
        <div>
          <div className="card-label">Stage</div>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, color: 'var(--normal)' }}>{model.stage}</div>
        </div>
        <div>
          <div className="card-label">Validation Loss</div>
          <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>{model.val_loss?.toFixed(6)}</div>
        </div>
        <div>
          <div className="card-label">Anomaly Threshold</div>
          <div style={{ fontFamily: 'var(--font-mono)', color: 'var(--medium)' }}>{model.anomaly_threshold?.toFixed(6)}</div>
        </div>
        <div>
          <div className="card-label">Run ID</div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.65rem', color: 'var(--text-muted)', wordBreak: 'break-all' }}>
            {model.run_id?.slice(0, 16)}…
          </div>
        </div>
      </div>
    </div>
  );
}

function DagDetails({ dag, title }) {
  if (!dag) return null;
  const duration = dag.duration_seconds != null ? `${dag.duration_seconds.toFixed(1)}s` : '—';
  const stateColor = { success: 'var(--normal)', failed: 'var(--high)', running: 'var(--cyan)' };

  return (
    <div className="card">
      <div className="section-header">
        <span className="section-title">{title}</span>
        <span style={{ fontSize: '0.72rem', color: stateColor[dag.state] ?? 'var(--text-secondary)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          {dag.state}
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 12, fontSize: '0.75rem' }}>
        <div>
          <div className="card-label">Run ID</div>
          <div style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: '0.65rem', wordBreak: 'break-all' }}>{dag.run_id}</div>
        </div>
        <div>
          <div className="card-label">Duration</div>
          <div style={{ color: 'var(--text-primary)' }}>{duration}</div>
        </div>
        <div>
          <div className="card-label">Started</div>
          <div style={{ color: 'var(--text-secondary)' }}>{dag.start_date ? new Date(dag.start_date).toLocaleString() : '—'}</div>
        </div>
        <div>
          <div className="card-label">Finished</div>
          <div style={{ color: 'var(--text-secondary)' }}>{dag.end_date ? new Date(dag.end_date).toLocaleString() : '—'}</div>
        </div>
      </div>
    </div>
  );
}

export default function PipelinePage() {
  const { data: status, loading, error, refresh } = usePolling(fetchPipelineStatus, 15000);

  return (
    <div className="page-enter">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ marginBottom: 4 }}>ML Pipeline</h1>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
            AIRFLOW DAGS · MLFLOW REGISTRY · DVC VERSIONING
          </div>
        </div>
        <button
          onClick={refresh}
          style={{
            background: 'var(--bg-elevated)', border: '1px solid var(--border-bright)',
            borderRadius: 'var(--r-md)', color: 'var(--text-secondary)',
            padding: '6px 14px', cursor: 'pointer', fontSize: '0.72rem',
            fontFamily: 'var(--font-mono)', letterSpacing: '0.06em',
          }}
        >
          ↻ Refresh
        </button>
      </div>

      {/* ── Summary KPIs ─────────────────────────────────────────────── */}
      <div className="grid-3" style={{ marginBottom: 20 }}>
        <StatCard
          label="Total MLflow Runs"
          value={status?.total_runs ?? '—'}
          color="var(--cyan)"
          sub={`Experiment: ${status?.mlflow_experiment ?? '—'}`}
        />
        <StatCard
          label="Ingestion DAG State"
          value={status?.ingestion_dag?.state ?? 'unknown'}
          color={status?.ingestion_dag?.state === 'success' ? 'var(--normal)' : 'var(--medium)'}
          sub={`Run: ${(status?.ingestion_dag?.run_id ?? '—').slice(0,20)}`}
        />
        <StatCard
          label="Last Retrain"
          value={status?.training_dag?.start_date
            ? new Date(status.training_dag.start_date).toLocaleDateString()
            : 'Never'}
          color="var(--text-primary)"
          sub={`State: ${status?.training_dag?.state ?? '—'}`}
        />
      </div>

      {/* ── Pipeline DAG visualisation ───────────────────────────────── */}
      {loading && !status && <div className="loader">Loading pipeline status</div>}
      {error && (
        <div className="card" style={{ color: 'var(--medium)', marginBottom: 16 }}>
          ⚠ Could not reach Airflow or MLflow — check services are running.<br/>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{error}</span>
        </div>
      )}
      <PipelineViz status={status} />

      {/* ── DAG run details ──────────────────────────────────────────── */}
      <div className="grid-2" style={{ marginTop: 16 }}>
        <DagDetails dag={status?.ingestion_dag} title="Ingestion DAG — Latest Run" />
        <DagDetails dag={status?.training_dag}  title="Training DAG — Latest Run" />
      </div>

      {/* ── Model info ───────────────────────────────────────────────── */}
      <div style={{ marginTop: 16 }}>
        <ModelCard model={status?.current_model} />
      </div>

      {/* ── Quick links ──────────────────────────────────────────────── */}
      <div className="card" style={{ marginTop: 16 }}>
        <div className="section-title" style={{ marginBottom: 14 }}>External Consoles</div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Airflow UI',    url: 'http://localhost:8080', color: 'var(--cyan)' },
            { label: 'MLflow UI',     url: 'http://localhost:5000', color: 'var(--normal)' },
            { label: 'Grafana',       url: 'http://localhost:3001', color: 'var(--medium)' },
            { label: 'Prometheus',    url: 'http://localhost:9090', color: 'var(--low)' },
            { label: 'API Docs',      url: 'http://localhost:8000/docs', color: 'var(--text-secondary)' },
          ].map(link => (
            <a
              key={link.label}
              href={link.url}
              target="_blank"
              rel="noreferrer"
              style={{
                display: 'inline-block',
                padding: '6px 16px',
                border: `1px solid ${link.color}`,
                borderRadius: 'var(--r-md)',
                color: link.color,
                fontSize: '0.72rem',
                fontFamily: 'var(--font-mono)',
                textDecoration: 'none',
                letterSpacing: '0.06em',
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.target.style.background = `${link.color}18`}
              onMouseLeave={e => e.target.style.background = 'transparent'}
            >
              ↗ {link.label}
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
