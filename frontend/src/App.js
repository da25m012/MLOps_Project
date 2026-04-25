import React from 'react';
import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import PipelinePage from './pages/PipelinePage';
import DriftPage from './pages/DriftPage';
import SettingsPage from './pages/SettingsPage';

// ── Icons (inline SVG, no icon-font dep) ──────────────────────────────────
const IconDashboard = () => (
  <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <rect x="1" y="1" width="6" height="6" rx="1"/>
    <rect x="9" y="1" width="6" height="6" rx="1"/>
    <rect x="1" y="9" width="6" height="6" rx="1"/>
    <rect x="9" y="9" width="6" height="6" rx="1"/>
  </svg>
);

const IconPipeline = () => (
  <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="2" cy="8" r="1.5"/>
    <circle cx="8" cy="8" r="1.5"/>
    <circle cx="14" cy="8" r="1.5"/>
    <line x1="3.5" y1="8" x2="6.5" y2="8"/>
    <line x1="9.5" y1="8" x2="12.5" y2="8"/>
    <polyline points="4,5 8,2 12,5"/>
    <polyline points="4,11 8,14 12,11"/>
  </svg>
);

const IconDrift = () => (
  <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <polyline points="1,12 4,7 7,9 10,4 13,6 15,3"/>
    <line x1="1" y1="14" x2="15" y2="14"/>
  </svg>
);

const IconSettings = () => (
  <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
    <circle cx="8" cy="8" r="2.5"/>
    <path d="M8 1v2M8 13v2M1 8h2M13 8h2M3.05 3.05l1.41 1.41M11.54 11.54l1.41 1.41M3.05 12.95l1.41-1.41M11.54 4.46l1.41-1.41"/>
  </svg>
);

function Sidebar() {
  const location = useLocation();

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="logo-mark">AnomalyOS</span>
        <span className="logo-sub">Detection System v1.0</span>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/" end className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <IconDashboard /> Dashboard
        </NavLink>
        <NavLink to="/pipeline" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <IconPipeline /> ML Pipeline
        </NavLink>
        <NavLink to="/drift" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <IconDrift /> Drift Report
        </NavLink>
        <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <IconSettings /> Settings
        </NavLink>
      </nav>

      <div className="sidebar-footer">
        LSTM Autoencoder · PyTorch 2.2<br/>
        MLflow · Airflow · Prometheus
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <Sidebar />
        <main className="main-content">
          <Routes>
            <Route path="/"         element={<Dashboard />} />
            <Route path="/pipeline" element={<PipelinePage />} />
            <Route path="/drift"    element={<DriftPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
