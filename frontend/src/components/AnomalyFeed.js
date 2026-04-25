// components/AnomalyFeed.js
import React from 'react';
import { format } from 'date-fns';
import SeverityBadge from './SeverityBadge';

export default function AnomalyFeed({ entries = [] }) {
  const anomalies = entries.filter(e => e.is_anomaly).slice(-20).reverse();

  if (!anomalies.length) {
    return (
      <div style={{ padding: '24px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
        <div style={{ fontSize: '1.5rem', marginBottom: 8 }}>✓</div>
        No anomalies detected in this window
      </div>
    );
  }

  return (
    <div style={{ maxHeight: 320, overflowY: 'auto' }}>
      {anomalies.map((entry, i) => (
        <div key={i} className="feed-item">
          <div className="feed-time">
            {entry.timestamp ? format(new Date(entry.timestamp), 'HH:mm:ss') : '—'}
          </div>
          <div className="feed-body">
            <div className="feed-title">
              Reconstruction error:{' '}
              <span style={{ color: 'var(--high)', fontWeight: 700 }}>
                {entry.reconstruction_error?.toFixed(5)}
              </span>
            </div>
            <div className="feed-meta">
              CPU: {entry.cpu_percent?.toFixed(1)}% · Mem: {entry.mem_percent?.toFixed(1)}% · Err rate: {entry.error_rate_percent?.toFixed(2)}%
            </div>
          </div>
          <SeverityBadge severity={entry.severity} pulse={entry.severity >= 2} />
        </div>
      ))}
    </div>
  );
}
