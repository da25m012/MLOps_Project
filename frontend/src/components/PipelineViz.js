// components/PipelineViz.js
import React from 'react';

const INGESTION_STAGES = [
  { id: 'collect',    label: 'Collect' },
  { id: 'validate',  label: 'Validate' },
  { id: 'baselines', label: 'Baselines' },
  { id: 'preprocess',label: 'Preprocess' },
  { id: 'dvc',       label: 'DVC Commit' },
];

const TRAINING_STAGES = [
  { id: 'load',     label: 'Load Feat.' },
  { id: 'train',    label: 'Train LSTM' },
  { id: 'evaluate', label: 'Evaluate' },
  { id: 'promote',  label: 'Promote' },
  { id: 'notify',   label: 'Notify' },
];

function stateClass(dagState, idx, total) {
  if (!dagState) return 'pending';
  if (dagState === 'success') return 'success';
  if (dagState === 'failed')  return idx === 0 ? 'failed' : 'pending';
  if (dagState === 'running') {
    // Simulate progress through stages
    const progress = Math.floor(total * 0.6);
    if (idx < progress) return 'success';
    if (idx === progress) return 'running';
    return 'pending';
  }
  return 'pending';
}

function stateSymbol(cls) {
  if (cls === 'success') return '✓';
  if (cls === 'failed')  return '✗';
  if (cls === 'running') return '◉';
  return '○';
}

function DAGRow({ title, stages, dagState, lastRun }) {
  return (
    <div style={{ marginBottom: 28 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <span className="section-title">{title}</span>
        {lastRun && (
          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>
            Last run: <span style={{ color: 'var(--text-secondary)' }}>{lastRun}</span>
          </span>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'center' }}>
        {stages.map((stage, idx) => {
          const cls = stateClass(dagState, idx, stages.length);
          return (
            <React.Fragment key={stage.id}>
              <div className="pipeline-stage">
                <div className={`stage-node ${cls}`}>{stateSymbol(cls)}</div>
                <div className="stage-label">{stage.label}</div>
              </div>
              {idx < stages.length - 1 && (
                <div
                  className={`pipeline-connector ${stateClass(dagState, idx, stages.length) === 'success' ? 'active' : ''}`}
                />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}

export default function PipelineViz({ status }) {
  const ingestion = status?.ingestion_dag;
  const training  = status?.training_dag;

  const fmt = (dt) => dt ? new Date(dt).toLocaleString() : 'Never';

  return (
    <div className="card">
      <DAGRow
        title="Ingestion DAG — metric_ingestion_pipeline"
        stages={INGESTION_STAGES}
        dagState={ingestion?.state}
        lastRun={fmt(ingestion?.start_date)}
      />
      <div style={{ borderTop: '1px solid var(--border)', margin: '4px 0 24px' }} />
      <DAGRow
        title="Training DAG — model_training_pipeline"
        stages={TRAINING_STAGES}
        dagState={training?.state}
        lastRun={fmt(training?.start_date)}
      />
    </div>
  );
}
