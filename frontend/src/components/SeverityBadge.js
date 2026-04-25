// components/SeverityBadge.js
import React from 'react';

const SEVERITY_MAP = {
  0: { label: 'Normal', cls: 'badge-normal' },
  1: { label: 'Low',    cls: 'badge-low' },
  2: { label: 'Medium', cls: 'badge-medium' },
  3: { label: 'High',   cls: 'badge-high' },
};

export default function SeverityBadge({ severity, pulse = false }) {
  const { label, cls } = SEVERITY_MAP[severity] ?? SEVERITY_MAP[0];
  return (
    <span className={`badge ${cls}`}>
      {pulse && <span className="badge-dot" />}
      {label}
    </span>
  );
}
