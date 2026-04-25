// components/StatCard.js
import React from 'react';

export default function StatCard({ label, value, unit = '', color = 'var(--cyan)', sub }) {
  return (
    <div className="card">
      <div className="card-label">{label}</div>
      <div className="card-value" style={{ color }}>
        {value ?? '—'}
        {unit && <span style={{ fontSize: '1rem', fontWeight: 400, color: 'var(--text-secondary)', marginLeft: 4 }}>{unit}</span>}
      </div>
      {sub && <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 6 }}>{sub}</div>}
    </div>
  );
}
