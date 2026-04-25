// pages/DriftPage.js
import React from 'react';
import { usePolling } from '../hooks/usePolling';
import { fetchDriftReport } from '../services/api';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine } from 'recharts';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-bright)', borderRadius: 8, padding: '10px 14px', fontSize: '0.72rem' }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>{d.feature}</div>
      <div>PSI: <strong style={{ color: d.drift_detected ? 'var(--high)' : 'var(--normal)' }}>{d.psi_score}</strong></div>
      <div style={{ color: 'var(--text-muted)', marginTop: 4 }}>
        Ref mean: {d.reference_mean} → Cur mean: {d.current_mean}
      </div>
    </div>
  );
};

export default function DriftPage() {
  const { data, loading, error, refresh } = usePolling(fetchDriftReport, 30000);

  return (
    <div className="page-enter">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <h1 style={{ marginBottom: 4 }}>Data Drift Report</h1>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
            PSI-BASED FEATURE DISTRIBUTION MONITORING
          </div>
        </div>
        <button
          onClick={refresh}
          style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-bright)', borderRadius: 'var(--r-md)', color: 'var(--text-secondary)', padding: '6px 14px', cursor: 'pointer', fontSize: '0.72rem', fontFamily: 'var(--font-mono)' }}
        >
          ↻ Refresh
        </button>
      </div>

      {loading && !data && <div className="loader">Computing drift scores</div>}
      {error && (
        <div className="card" style={{ color: 'var(--medium)', marginBottom: 16 }}>
          ⚠ Drift service unavailable — baseline files may not exist yet.<br/>
          <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Run the ingestion pipeline first to generate baselines.</span>
        </div>
      )}

      {data && (
        <>
          {/* ── Overall status ─────────────────────────────────────── */}
          <div className="card" style={{
            marginBottom: 20,
            borderColor: data.overall_drift_detected ? 'var(--high)' : 'var(--normal)',
            background: data.overall_drift_detected ? 'rgba(255,56,96,0.05)' : 'rgba(0,255,157,0.04)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <div style={{ fontSize: '2rem' }}>{data.overall_drift_detected ? '⚠' : '✓'}</div>
              <div>
                <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '1.1rem', color: data.overall_drift_detected ? 'var(--high)' : 'var(--normal)' }}>
                  {data.overall_drift_detected ? 'DRIFT DETECTED' : 'NO DRIFT DETECTED'}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                  {data.recommendation}
                </div>
                <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 6 }}>
                  PSI threshold: {data.psi_threshold} · Generated: {data.generated_at ? new Date(data.generated_at).toLocaleString() : '—'}
                </div>
              </div>
            </div>
          </div>

          {/* ── PSI bar chart ──────────────────────────────────────── */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="section-title" style={{ marginBottom: 16 }}>PSI Score per Feature</div>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={data.feature_stats} margin={{ left: -20, right: 8 }}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
                <XAxis dataKey="feature" tick={{ fill: 'var(--text-muted)', fontSize: 9, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} angle={-20} textAnchor="end" height={48} />
                <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <ReferenceLine y={data.psi_threshold} stroke="var(--high)" strokeDasharray="6 3" label={{ value: 'threshold', fill: 'var(--high)', fontSize: 10, position: 'insideTopRight' }} />
                <Bar dataKey="psi_score" radius={[3,3,0,0]}
                  fill="var(--cyan)"
                  label={false}
                  isAnimationActive={false}
                  // Color each bar by drift status
                  cells={data.feature_stats?.map((entry, index) => (
                    <cell key={index} fill={entry.drift_detected ? 'var(--high)' : 'var(--cyan)'} />
                  ))}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* ── Feature table ──────────────────────────────────────── */}
          <div className="card">
            <div className="section-title" style={{ marginBottom: 14 }}>Feature Detail</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                  {['Feature', 'PSI', 'Ref Mean', 'Cur Mean', 'Ref Std', 'Cur Std', 'Drift'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 12px 10px', borderBottom: '1px solid var(--border)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.feature_stats?.map(f => (
                  <tr key={f.feature} style={{ borderBottom: '1px solid var(--border)', transition: 'background 0.15s' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-hover)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    <td style={{ padding: '8px 12px', color: 'var(--text-primary)' }}>{f.feature}</td>
                    <td style={{ padding: '8px 12px', color: f.drift_detected ? 'var(--high)' : 'var(--normal)', fontWeight: 700 }}>{f.psi_score}</td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-secondary)' }}>{f.reference_mean}</td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-secondary)' }}>{f.current_mean}</td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>{f.reference_std}</td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>{f.current_std}</td>
                    <td style={{ padding: '8px 12px' }}>
                      {f.drift_detected
                        ? <span style={{ color: 'var(--high)', fontWeight: 700 }}>YES</span>
                        : <span style={{ color: 'var(--normal)' }}>NO</span>
                      }
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
