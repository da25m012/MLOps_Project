// components/FeatureHeatmap.js
import React from 'react';

const FEATURES = [
  'cpu_percent', 'mem_percent', 'disk_io_read_mb', 'disk_io_write_mb',
  'net_bytes_sent_mb', 'net_bytes_recv_mb', 'req_rate_per_sec', 'error_rate_percent',
];

function errColor(val, max) {
  if (!max || max === 0) return 'var(--bg-elevated)';
  const ratio = Math.min(val / max, 1);
  if (ratio < 0.3) return `rgba(0,255,157,${0.1 + ratio})`;
  if (ratio < 0.6) return `rgba(255,224,102,${0.1 + ratio})`;
  if (ratio < 0.85) return `rgba(255,159,67,${0.1 + ratio})`;
  return `rgba(255,56,96,${0.2 + ratio * 0.6})`;
}

export default function FeatureHeatmap({ perFeatureErrors = [] }) {
  const max = Math.max(...perFeatureErrors, 0.0001);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>
      {FEATURES.map((f, i) => {
        const val = perFeatureErrors[i] ?? 0;
        return (
          <div
            key={f}
            style={{
              background: errColor(val, max),
              border: '1px solid var(--border)',
              borderRadius: 'var(--r-sm)',
              padding: '10px 8px',
              textAlign: 'center',
              transition: 'background 0.4s',
            }}
          >
            <div style={{ fontSize: '0.6rem', color: 'var(--text-muted)', letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 4 }}>
              {f.replace(/_/g, ' ')}
            </div>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
              {val.toFixed(4)}
            </div>
          </div>
        );
      })}
    </div>
  );
}
