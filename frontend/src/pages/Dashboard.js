// pages/Dashboard.js
import React, { useMemo } from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchMetricsHistory, fetchMetricsSummary, fetchReady } from '../services/api';
import StatCard from '../components/StatCard';
import MetricsChart from '../components/MetricsChart';
import AnomalyFeed from '../components/AnomalyFeed';
import FeatureHeatmap from '../components/FeatureHeatmap';
import SeverityBadge from '../components/SeverityBadge';

export default function Dashboard() {
  const { data: history } = usePolling(() => fetchMetricsHistory(30, 300), 5000);
  const { data: summary } = usePolling(fetchMetricsSummary, 5000);
  const { data: ready }   = usePolling(fetchReady, 10000);

  const entries = history?.entries ?? [];
  const latest  = entries[entries.length - 1];

  // Anomaly rate over last 5 min window
  const anomalyRate = useMemo(() => {
    if (!entries.length) return 0;
    const recent = entries.slice(-60);
    const anomalous = recent.filter(e => e.is_anomaly).length;
    return ((anomalous / recent.length) * 100).toFixed(1);
  }, [entries]);

  const modelLoaded = ready?.model_loaded ?? false;

  return (
    <div className="page-enter">
      {/* ── Page header ──────────────────────────────────────────────── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ marginBottom: 4 }}>System Dashboard</h1>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
            REAL-TIME ANOMALY DETECTION · UPDATING EVERY 5s
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: '0.7rem', color: modelLoaded ? 'var(--normal)' : 'var(--high)' }}>
            {modelLoaded ? '● MODEL ONLINE' : '○ MODEL OFFLINE'}
          </span>
          {latest && <SeverityBadge severity={latest.severity ?? 0} pulse={latest.is_anomaly} />}
        </div>
      </div>

      {/* ── KPI row ─────────────────────────────────────────────────── */}
      <div className="grid-4" style={{ marginBottom: 20 }}>
        <StatCard
          label="Current CPU"
          value={latest?.cpu_percent?.toFixed(1) ?? '—'}
          unit="%"
          color="var(--cyan)"
          sub={`Mem: ${latest?.mem_percent?.toFixed(1) ?? '—'}%`}
        />
        <StatCard
          label="Error Rate"
          value={latest?.error_rate_percent?.toFixed(2) ?? '—'}
          unit="%"
          color={parseFloat(latest?.error_rate_percent) > 5 ? 'var(--high)' : 'var(--normal)'}
          sub="Threshold: 5%"
        />
        <StatCard
          label="Reconstruction Error"
          value={latest?.reconstruction_error?.toFixed(5) ?? '—'}
          color="var(--medium)"
          sub={`Threshold: ${summary?.threshold?.toFixed(5) ?? '—'}`}
        />
        <StatCard
          label="Anomaly Rate (5m)"
          value={anomalyRate}
          unit="%"
          color={parseFloat(anomalyRate) > 10 ? 'var(--high)' : 'var(--normal)'}
          sub={`${entries.filter(e => e.is_anomaly).length} total flagged`}
        />
      </div>

      {/* ── System metrics chart ─────────────────────────────────────── */}
      <MetricsChart
        data={entries}
        series={['cpu_percent', 'mem_percent', 'error_rate_percent']}
        height={200}
        title="System Metrics"
      />

      {/* ── Anomaly score chart ──────────────────────────────────────── */}
      <MetricsChart
        data={entries}
        series={['reconstruction_error']}
        threshold={summary?.threshold}
        height={160}
        title="Reconstruction Error (Anomaly Score)"
      />

      {/* ── Request rate chart ───────────────────────────────────────── */}
      <MetricsChart
        data={entries}
        series={['req_rate_per_sec']}
        height={140}
        title="Request Rate"
      />

      {/* ── Bottom row: feed + heatmap ───────────────────────────────── */}
      <div className="grid-2" style={{ marginTop: 4 }}>
        <div className="card">
          <div className="section-header">
            <span className="section-title">Anomaly Event Feed</span>
            <span className="badge badge-info">{entries.filter(e => e.is_anomaly).length} events</span>
          </div>
          <AnomalyFeed entries={entries} />
        </div>

        <div className="card">
          <div className="section-header">
            <span className="section-title">Per-Feature Error Heatmap</span>
          </div>
          <FeatureHeatmap perFeatureErrors={latest?.per_feature_errors ?? []} />
          <div style={{ marginTop: 16, fontSize: '0.68rem', color: 'var(--text-muted)', lineHeight: 1.7 }}>
            <span style={{ color: 'var(--normal)' }}>■</span> Normal &nbsp;
            <span style={{ color: 'var(--low)' }}>■</span> Low &nbsp;
            <span style={{ color: 'var(--medium)' }}>■</span> Medium &nbsp;
            <span style={{ color: 'var(--high)' }}>■</span> High
          </div>
        </div>
      </div>
    </div>
  );
}
