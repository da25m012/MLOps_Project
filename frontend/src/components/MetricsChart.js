// components/MetricsChart.js
import React from 'react';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, ReferenceLine, CartesianGrid, Legend,
} from 'recharts';
import { format } from 'date-fns';

const COLORS = {
  cpu_percent:         '#00e5ff',
  mem_percent:         '#b39ddb',
  error_rate_percent:  '#ff3860',
  req_rate_per_sec:    '#00ff9d',
  reconstruction_error:'#ff9f43',
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border-bright)',
      borderRadius: 'var(--r-md)',
      padding: '10px 14px',
      fontSize: '0.72rem',
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toFixed(3) : p.value}</strong>
        </div>
      ))}
    </div>
  );
};

export default function MetricsChart({ data = [], series = [], threshold, height = 200, title }) {
  const formatted = data.map(d => ({
    ...d,
    _ts: d.timestamp ? format(new Date(d.timestamp), 'HH:mm:ss') : '',
  }));

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      {title && (
        <div className="section-header" style={{ marginBottom: 12 }}>
          <span className="section-title">{title}</span>
        </div>
      )}
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={formatted} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="4 4" vertical={false} />
            <XAxis
              dataKey="_ts"
              tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              axisLine={false} tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
              axisLine={false} tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', paddingTop: 8 }}
            />
            {threshold !== undefined && (
              <ReferenceLine
                y={threshold} stroke="var(--high)" strokeDasharray="6 3"
                label={{ value: 'threshold', fill: 'var(--high)', fontSize: 10, position: 'insideTopRight' }}
              />
            )}
            {series.map(key => (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={COLORS[key] || 'var(--cyan)'}
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
                name={key.replace(/_/g, ' ')}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
