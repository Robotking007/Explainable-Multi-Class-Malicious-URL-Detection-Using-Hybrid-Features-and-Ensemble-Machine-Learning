import React, { useState } from 'react';
import {
  Settings as SettingsIcon,
  Server,
  RefreshCw,
  Cpu,
  Shield,
  Layers,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Database,
  Terminal,
  ExternalLink,
  Save,
  Check
} from 'lucide-react';
import { useBackend } from '../context/BackendContext';

export const Settings = () => {
  const {
    status,
    isOnline,
    latency,
    lastChecked,
    backendMeta,
    apiUrl,
    updateApiUrl,
    refreshStatus,
    moduleStatus
  } = useBackend();

  const [inputUrl, setInputUrl] = useState(apiUrl);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleSaveUrl = (e) => {
    e.preventDefault();
    if (!inputUrl.trim()) return;
    updateApiUrl(inputUrl.trim());
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2500);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">SYSTEM ORCHESTRATION</span>
            <span className="text-xs text-slate-400 font-mono">• Configuration & Health</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            Settings & System Status
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Configure backend FastAPI server connections, review active deep learning checkpoints, and audit client telemetry.
          </p>
        </div>

        <button
          onClick={refreshStatus}
          className="soc-btn-secondary text-xs px-4 py-2 font-mono flex items-center gap-2"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${status === 'checking' ? 'animate-spin' : ''}`} />
          <span>Ping Backend Now</span>
        </button>
      </div>

      {/* Grid: FastAPI Connection Setup & Live Health */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Backend API Configuration */}
        <div className="soc-card p-6 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold font-mono text-slate-100">FastAPI Server Endpoint</h3>
            </div>
            <span
              className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold ${
                isOnline
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                  : 'bg-rose-500/10 text-rose-400 border border-rose-500/30'
              }`}
            >
              {isOnline ? 'CONNECTED' : 'OFFLINE'}
            </span>
          </div>

          <form onSubmit={handleSaveUrl} className="space-y-3 font-mono text-xs">
            <label className="block text-slate-300 text-xs">Base API URL (Uvicorn Host)</label>
            <div className="flex gap-2">
              <input
                type="text"
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                placeholder="http://127.0.0.1:8000"
                className="flex-1 bg-[#080d19] border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-slate-100 focus:outline-none focus:border-cyan-400 font-mono"
              />
              <button
                type="submit"
                className="soc-btn-primary px-4 py-2.5 text-xs font-mono flex items-center gap-1.5"
              >
                {saveSuccess ? <Check className="w-4 h-4 text-slate-950" /> : <Save className="w-4 h-4" />}
                <span>Apply</span>
              </button>
            </div>
            <p className="text-[11px] text-slate-400 font-sans">
              Environment fallback: <code className="text-cyan-300">VITE_API_BASE_URL</code>. Changes take effect immediately across all service modules.
            </p>
          </form>

          {/* Connection Diagnostics */}
          <div className="pt-3 border-t border-slate-800 space-y-2 font-mono text-xs">
            <div className="flex justify-between p-2.5 rounded bg-slate-900 border border-slate-800">
              <span className="text-slate-400">Connection Status:</span>
              <span className={isOnline ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                {status.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between p-2.5 rounded bg-slate-900 border border-slate-800">
              <span className="text-slate-400">Round-Trip Latency (RTT):</span>
              <span className="text-slate-200">{latency !== null ? `${latency} ms` : 'N/A'}</span>
            </div>
            <div className="flex justify-between p-2.5 rounded bg-slate-900 border border-slate-800">
              <span className="text-slate-400">Last Health Probe:</span>
              <span className="text-slate-200">
                {lastChecked ? lastChecked.toLocaleTimeString() : 'Never'}
              </span>
            </div>
          </div>
        </div>

        {/* Backend Spec / Health Payload */}
        <div className="soc-card p-6 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-violet-400" />
              <h3 className="text-sm font-bold font-mono text-slate-100">Live API Response Data</h3>
            </div>
            <span className="text-[11px] font-mono text-slate-400">REST Contract</span>
          </div>

          <div className="bg-[#080d19] border border-slate-800 rounded-xl p-4 font-mono text-xs text-slate-300 h-52 overflow-y-auto">
            {backendMeta ? (
              <pre className="text-[11px] text-cyan-300 whitespace-pre-wrap">
                {JSON.stringify(backendMeta, null, 2)}
              </pre>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-slate-500 text-center">
                <Terminal className="w-6 h-6 mb-2" />
                <p>Waiting for successful response from {apiUrl}...</p>
                <p className="text-[10px] text-slate-600 mt-1">Start backend with: uvicorn main:app --reload</p>
              </div>
            )}
          </div>

          <div className="text-[11px] text-slate-400 font-sans">
            Health checks verify root <code className="text-cyan-300">/health</code> or <code className="text-cyan-300">/api/health</code> endpoint responsiveness.
          </div>
        </div>
      </div>

      {/* Model & Architecture Status */}
      <div className="soc-card p-6 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold font-mono text-slate-100">
              Registered Deep Learning Modules
            </h3>
          </div>
          <span className="text-xs font-mono text-slate-400">4 Core Models</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 font-mono text-xs">
          {Object.entries(moduleStatus).map(([key, mod]) => (
            <div key={key} className="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-200">{mod.name}</span>
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
              </div>
              <p className="text-[11px] text-slate-400 font-sans">
                {key === 'urlGNN'
                  ? 'Spatial Graph Neural Network for URL lexical & infrastructure analysis.'
                  : key === 'malmem'
                  ? 'Volatile memory dump classifier trained on CIC-MalMem-2022.'
                  : key === 'bodmas'
                  ? 'Static PE header and entropy model trained on BODMAS & EMBER-2018.'
                  : key === 'malapi'
                  ? 'Sequential API call graph model trained on MalAPI2019.'
                  : key === 'xaiEngine'
                  ? 'Attribution engine computing Integrated Gradients and TreeSHAP.'
                  : 'Isolated headless container for controlled dynamic inspection.'}
              </p>
              <div className="pt-2 border-t border-slate-800 flex justify-between text-[10px] text-slate-500">
                <span>STATUS: {mod.status?.toUpperCase()}</span>
                <span className="text-cyan-400">READY</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Application Meta */}
      <div className="soc-card p-6 space-y-2 font-mono text-xs">
        <h4 className="font-bold text-slate-200">Project Metadata</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 text-slate-400">
          <div>
            <span className="block text-[10px] text-slate-500">PROJECT</span>
            <span className="text-slate-300">EDAI Capstone Research Platform</span>
          </div>
          <div>
            <span className="block text-[10px] text-slate-500">FRONTEND VERSION</span>
            <span className="text-cyan-300">CyberX UI v1.0.0 (React 19 + Tailwind CSS)</span>
          </div>
          <div>
            <span className="block text-[10px] text-slate-500">BACKEND STACK</span>
            <span className="text-slate-300">FastAPI + Uvicorn + PyTorch GNN</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
