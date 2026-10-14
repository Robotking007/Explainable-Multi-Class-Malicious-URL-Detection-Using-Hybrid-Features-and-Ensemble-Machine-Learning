import React, { useState } from 'react';
import {
  Cpu,
  Globe,
  Search,
  CheckCircle2,
  Clock,
  AlertTriangle,
  Code2,
  ShieldCheck,
  ShieldAlert,
  Server,
  Layers,
  Terminal,
  ExternalLink,
  Info
} from 'lucide-react';
import dynamicAnalysisService from '../services/dynamicAnalysisService';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorBanner from '../components/common/ErrorBanner';
import StatusBadge from '../components/common/StatusBadge';

export const DynamicAnalysis = () => {
  const [urlInput, setUrlInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dynamicResult, setDynamicResult] = useState(null);

  const stagesWorkflow = [
    { id: 1, title: 'Static Lexical Parser', desc: 'Entropy, URL length, token n-grams' },
    { id: 2, title: 'Confidence Check', desc: 'Threshold gating before heavy sandbox' },
    { id: 3, title: 'Headless Browser Sandbox', desc: 'Controlled isolated container execution' },
    { id: 4, title: 'DOM & JavaScript Profiler', desc: 'Forms, hidden elements, event listeners' },
    { id: 5, title: 'Multi-Modal GNN Verdict', desc: 'Unified static + dynamic inference' }
  ];

  const handleStartAnalysis = async (url = urlInput) => {
    if (!url || !url.trim()) {
      setError('Please provide a URL to dynamically inspect.');
      return;
    }

    setLoading(true);
    setError(null);
    setDynamicResult(null);

    try {
      const res = await dynamicAnalysisService.runDynamicAnalysis(url.trim());
      setDynamicResult(res);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">SANDBOX ENGINE</span>
            <span className="text-xs text-slate-400 font-mono">• Controlled Execution</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            Dynamic URL Analysis
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Multi-stage workflow: Static Features → Confidence Evaluation → Headless Browser Sandbox → Behavioral DOM & JS Profiling.
          </p>
        </div>
      </div>

      {/* Isolation Safety Disclaimer */}
      <div className="p-4 rounded-xl bg-slate-900/80 border border-cyan-500/20 text-xs text-slate-300 flex items-start gap-3">
        <Info className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-semibold text-slate-200 font-mono">Sandbox Isolation Notice: </span>
          <span>
            URLs are strictly executed within the backend's controlled container environment with network egress restrictions.
            The frontend never directly renders or executes foreign untrusted JavaScript in your local browser session.
          </span>
        </div>
      </div>

      {/* Target Input Card */}
      <div className="soc-card p-5 space-y-3">
        <label htmlFor="dynamic-url-input" className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
          <Globe className="w-4 h-4 text-cyan-400" />
          <span>Target URL for Dynamic Sandbox Inspection</span>
        </label>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            id="dynamic-url-input"
            type="text"
            placeholder="https://suspicious-portal.site/verify?token=xyz"
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            disabled={loading}
            className="flex-1 bg-[#080d19] border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-400"
          />
          <button
            onClick={() => handleStartAnalysis()}
            disabled={loading || !urlInput.trim()}
            className="soc-btn-primary px-6 py-2.5 text-xs font-mono"
          >
            <Cpu className="w-4 h-4" />
            <span>Launch Dynamic Sandbox</span>
          </button>
        </div>
      </div>

      {/* Stepper / Timeline Workflow Visualizer */}
      <div className="soc-card p-6 space-y-4">
        <h3 className="text-xs font-bold font-mono text-slate-200 uppercase tracking-wider">
          Dynamic Analysis Execution Pipeline
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {stagesWorkflow.map((stage, idx) => (
            <div
              key={stage.id}
              className={`p-3.5 rounded-xl border font-mono text-xs transition-all ${
                dynamicResult
                  ? 'bg-slate-900/90 border-emerald-500/40 text-emerald-300'
                  : 'bg-slate-900/40 border-slate-800 text-slate-400'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 font-bold">
                  STAGE 0{stage.id}
                </span>
                {dynamicResult ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <Clock className="w-3.5 h-3.5 text-slate-600" />
                )}
              </div>
              <h4 className="font-bold text-slate-200 text-xs truncate">{stage.title}</h4>
              <p className="text-[10px] text-slate-400 mt-1 leading-tight">{stage.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {loading && (
        <LoadingSpinner
          message="Spawning Controlled Headless Browser Sandbox"
          subtext="Monitoring script hooks, credential harvesting forms, and WebSocket connections..."
          step="Stage 3: Headless Sandbox"
        />
      )}

      {error && (
        <ErrorBanner
          error={error}
          title="Sandbox Analysis Failed"
          onRetry={() => handleStartAnalysis()}
        />
      )}

      {/* Dynamic Results Display */}
      {dynamicResult && !loading && (
        <div className="space-y-6">
          {/* Main Verdict Card */}
          <div className="soc-card p-6 border-l-4 border-l-cyan-400 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
              <div>
                <span className="text-[10px] font-mono text-slate-400">ANALYSIS TARGET</span>
                <h3 className="text-base font-bold font-mono text-cyan-300 break-all">{dynamicResult.url}</h3>
              </div>
              <StatusBadge status={dynamicResult.finalVerdict} size="lg" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* DOM Profile */}
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 font-mono text-xs space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <span className="font-bold text-slate-200">DOM Profile & Structure</span>
                  <Code2 className="w-4 h-4 text-cyan-400" />
                </div>
                <div className="space-y-2 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Detected HTML Forms:</span>
                    <span className="text-slate-200 font-bold">{dynamicResult.domProfile.formsCount}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Password Fields:</span>
                    <span className="text-rose-400 font-bold">{dynamicResult.domProfile.passwordFields}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Hidden iFrames:</span>
                    <span className="text-slate-200 font-bold">{dynamicResult.domProfile.hiddenIframes}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">External Script Sources:</span>
                    <span className="text-slate-200 font-bold">{dynamicResult.domProfile.externalScriptCount}</span>
                  </div>
                </div>
              </div>

              {/* JS Behavior Profiling */}
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 font-mono text-xs space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                  <span className="font-bold text-slate-200">JavaScript Execution Profiling</span>
                  <Terminal className="w-4 h-4 text-violet-400" />
                </div>
                <div className="space-y-2 text-[11px]">
                  <div className="flex justify-between">
                    <span className="text-slate-400">eval() / Obfuscation:</span>
                    <span className={dynamicResult.jsBehavior.evalUsage ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                      {dynamicResult.jsBehavior.evalUsage ? 'DETECTED' : 'None'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Anti-Debugging Hooks:</span>
                    <span className={dynamicResult.jsBehavior.antiDebuggingDetected ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                      {dynamicResult.jsBehavior.antiDebuggingDetected ? 'Active Evasion' : 'Clean'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">XHR / Fetch Calls:</span>
                    <span className="text-slate-200 font-bold">{dynamicResult.jsBehavior.networkCallsCount} requests</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Warnings list */}
            {dynamicResult.warnings?.length > 0 && (
              <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-500/25 space-y-2">
                <span className="text-xs font-bold font-mono text-amber-300 flex items-center gap-1.5">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  Behavioral Anomalies Observed in Sandbox:
                </span>
                <ul className="list-disc list-inside text-xs text-amber-200/90 font-mono space-y-1">
                  {dynamicResult.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default DynamicAnalysis;
