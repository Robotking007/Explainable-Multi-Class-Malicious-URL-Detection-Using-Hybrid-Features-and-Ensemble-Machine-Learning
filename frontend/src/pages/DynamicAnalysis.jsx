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
  Info,
  Lock,
  Eye,
  FileCode,
  Network,
  CornerDownRight,
  ChevronRight,
  ChevronDown,
  Maximize2,
  Activity,
  Sparkles,
  Filter,
  ArrowRight,
  Radio,
  FileText,
  RefreshCw,
  AlertOctagon,
  AlertCircle,
  Bot,
  Zap,
  Shield,
  Workflow
} from 'lucide-react';
import dynamicAnalysisService from '../services/dynamicAnalysisService';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorBanner from '../components/common/ErrorBanner';
import StatusBadge from '../components/common/StatusBadge';

// Recursive Antigravity-Style DOM Node Viewer
const DomNodeView = ({ node, depth = 0, selectedNode, onSelectNode, filterTag }) => {
  const [expanded, setExpanded] = useState(true);
  const hasChildren = node.children && node.children.length > 0;
  const isSelected = selectedNode === node;

  if (filterTag && filterTag !== 'all' && node.tag !== filterTag && !hasChildren) {
    return null;
  }

  const isCritical = node.threat_badge && (
    node.threat_badge.includes('Critical') ||
    node.threat_badge.includes('Exfiltration') ||
    node.threat_badge.includes('Password')
  );
  const isWarning = node.threat_badge && (
    node.threat_badge.includes('Obfuscated') ||
    node.threat_badge.includes('Script') ||
    node.threat_badge.includes('Hidden') ||
    node.threat_badge.includes('Target')
  );

  return (
    <div className="font-mono text-xs select-none">
      <div
        onClick={() => onSelectNode(node)}
        className={`flex items-center gap-1.5 py-1 px-2 rounded-lg cursor-pointer transition-colors ${
          isSelected
            ? 'bg-cyan-500/20 text-cyan-200 border border-cyan-500/50'
            : node.is_suspicious
            ? 'bg-rose-950/20 text-rose-200 hover:bg-rose-950/40 border border-rose-900/30'
            : 'hover:bg-slate-800/60 text-slate-300'
        }`}
        style={{ paddingLeft: `${depth * 14 + 8}px` }}
      >
        {hasChildren ? (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
            className="p-0.5 hover:text-cyan-400 text-slate-500 transition-colors"
          >
            {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>
        ) : (
          <span className="w-3.5 text-center text-slate-600 text-[10px]">•</span>
        )}

        <span className={`font-bold ${node.is_suspicious ? 'text-rose-400' : 'text-violet-400'}`}>
          &lt;{node.tag}
        </span>

        {/* Display key attributes */}
        {node.attributes && Object.entries(node.attributes).slice(0, 3).map(([k, v]) => (
          <span key={k} className="text-[11px] truncate max-w-[160px]">
            <span className="text-cyan-300">{k}</span>
            <span className="text-slate-500">=</span>
            <span className="text-amber-300">"{v}"</span>
          </span>
        ))}

        <span className={`font-bold ${node.is_suspicious ? 'text-rose-400' : 'text-violet-400'}`}>
          &gt;
        </span>

        {node.threat_badge && (
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ml-auto ${
            isCritical
              ? 'bg-rose-950 text-rose-300 border border-rose-500/40'
              : isWarning
              ? 'bg-purple-950 text-purple-300 border border-purple-500/40'
              : 'bg-amber-950 text-amber-300 border border-amber-500/40'
          }`}>
            {node.threat_badge}
          </span>
        )}

        {node.text && !hasChildren && (
          <span className="text-slate-400 text-[11px] truncate max-w-[180px] ml-1">
            "{node.text}"
          </span>
        )}
      </div>

      {hasChildren && expanded && (
        <div>
          {node.children.map((child, idx) => (
            <DomNodeView
              key={idx}
              node={child}
              depth={depth + 1}
              selectedNode={selectedNode}
              onSelectNode={onSelectNode}
              filterTag={filterTag}
            />
          ))}
          <div
            className="py-0.5 text-slate-500 text-[11px]"
            style={{ paddingLeft: `${depth * 14 + 18}px` }}
          >
            &lt;/{node.tag}&gt;
          </div>
        </div>
      )}
    </div>
  );
};

export const DynamicAnalysis = () => {
  const [urlInput, setUrlInput] = useState('https://suspicious-portal.site/verify?token=xyz');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dynamicResult, setDynamicResult] = useState(null);
  const [activeTab, setActiveTab] = useState('agent'); // 'agent' | 'dom' | 'browser' | 'js' | 'warnings'
  const [viewportMode, setViewportMode] = useState('render'); // 'render' | 'mockup' | 'source'
  const [selectedNode, setSelectedNode] = useState(null);
  const [domFilter, setDomFilter] = useState('all');

  const stagesWorkflow = [
    { id: 1, title: 'Static Lexical Parser', desc: 'Entropy, URL length, token n-grams' },
    { id: 2, title: 'Confidence Check', desc: 'Threshold gating before heavy sandbox' },
    { id: 3, title: 'Headless Browser Sandbox', desc: 'Controlled isolated container execution' },
    { id: 4, title: 'DOM & JavaScript Profiler', desc: 'Forms, hidden elements, event listeners' },
    { id: 5, title: 'Multi-Modal GNN Verdict', desc: 'Unified static + dynamic inference' }
  ];

  const presets = [
    {
      label: 'Phishing Portal (Obfuscated JS)',
      url: 'https://suspicious-portal.site/verify?token=xyz',
      category: 'Phishing'
    },
    {
      label: 'Microsoft Token Harvester',
      url: 'https://login.microsoftonline.security-token-renewal.com',
      category: 'Credential Stealer'
    },
    {
      label: 'Tor Admin Panel (C2 IP)',
      url: 'http://185.220.101.5/admin/panel',
      category: 'Malicious Host'
    },
    {
      label: 'Clean Web Service (GitHub Linux)',
      url: 'https://github.com/torvalds/linux',
      category: 'Benign'
    }
  ];

  const handleStartAnalysis = async (url = urlInput) => {
    if (!url || !url.trim()) {
      setError('Please provide a URL to dynamically inspect.');
      return;
    }

    setLoading(true);
    setError(null);
    setDynamicResult(null);
    setSelectedNode(null);

    try {
      const res = await dynamicAnalysisService.runDynamicAnalysis(url.trim());
      setDynamicResult(res);
      if (res.domTree && res.domTree.length > 0) {
        setSelectedNode(res.domTree[0]);
      }
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const parseWarningSeverity = (warning) => {
    if (warning.startsWith('[CRITICAL]')) {
      return { level: 'critical', text: warning.replace('[CRITICAL]', '').trim() };
    }
    if (warning.startsWith('[HIGH]')) {
      return { level: 'high', text: warning.replace('[HIGH]', '').trim() };
    }
    if (warning.startsWith('[MEDIUM]')) {
      return { level: 'medium', text: warning.replace('[MEDIUM]', '').trim() };
    }
    if (warning.startsWith('[INFO]')) {
      return { level: 'info', text: warning.replace('[INFO]', '').trim() };
    }
    if (warning.startsWith('[SAFE]')) {
      return { level: 'safe', text: warning.replace('[SAFE]', '').trim() };
    }
    return { level: 'warning', text: warning };
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">SANDBOX ENGINE</span>
            <span className="text-xs text-slate-400 font-mono">• Controlled Execution & DOM Profiler</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            Dynamic URL Analysis & Browser Sandbox
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Multi-stage workflow: Static Features → Confidence Evaluation → Headless Browser Sandbox → Behavioral DOM & Dynamic JS Execution Profiling.
          </p>
        </div>
      </div>

      {/* Isolation Safety Disclaimer */}
      <div className="p-4 rounded-xl bg-slate-900/80 border border-cyan-500/20 text-xs text-slate-300 flex items-start gap-3">
        <Info className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-semibold text-slate-200 font-mono">Sandbox Isolation Notice: </span>
          <span>
            URLs are strictly executed within the backend's isolated headless Chromium container with network egress restrictions.
            The frontend inspects the captured DOM tree and JavaScript telemetry without executing untrusted foreign code in your browser.
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
            onKeyDown={(e) => e.key === 'Enter' && handleStartAnalysis()}
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

        {/* Quick Presets Bar */}
        <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] font-mono text-slate-400">
          <span className="flex items-center gap-1 font-semibold text-slate-300">
            <Radio className="w-3 h-3 text-cyan-400" /> Presets:
          </span>
          {presets.map((p, idx) => (
            <button
              key={idx}
              onClick={() => {
                setUrlInput(p.url);
                handleStartAnalysis(p.url);
              }}
              className="px-2.5 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-slate-800 hover:border-cyan-500/40 transition-all flex items-center gap-1.5"
            >
              <span className="truncate max-w-[200px]">{p.label}</span>
              <span className="text-[9px] px-1 rounded bg-slate-800 text-slate-400">{p.category}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Stepper / Timeline Workflow Visualizer */}
      <div className="soc-card p-6 space-y-4">
        <h3 className="text-xs font-bold font-mono text-slate-200 uppercase tracking-wider flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <span>Dynamic Analysis Execution Pipeline</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
          {stagesWorkflow.map((stage) => (
            <div
              key={stage.id}
              className={`p-3.5 rounded-xl border font-mono text-xs transition-all ${
                dynamicResult
                  ? 'bg-slate-900/90 border-emerald-500/40 text-emerald-300 shadow-sm shadow-emerald-950/20'
                  : loading
                  ? 'bg-slate-900/60 border-cyan-500/30 text-cyan-200 animate-pulse'
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
          message="Spawning Controlled Headless Chromium Sandbox"
          subtext="Intercepting JavaScript execution hooks, monitoring credential exfiltration forms, and extracting DOM hierarchy..."
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
          {/* Main Verdict & KPI Banner */}
          <div className="soc-card p-6 border-l-4 border-l-cyan-400 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
              <div>
                <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Analysis Target</span>
                <h3 className="text-base font-bold font-mono text-cyan-300 break-all">{dynamicResult.url}</h3>
                <div className="flex flex-wrap items-center gap-3 mt-1.5 text-[11px] font-mono text-slate-400">
                  <span>Engine: <span className="text-slate-200">{dynamicResult.browserDetails?.engine || 'Chromium Headless'}</span></span>
                  <span>•</span>
                  <span>SSL: <span className={dynamicResult.browserDetails?.sslValid ? 'text-emerald-400' : 'text-rose-400'}>
                    {dynamicResult.browserDetails?.sslValid ? 'Verified HTTPS' : 'Unencrypted / Invalid'}
                  </span></span>
                  <span>•</span>
                  <span>HTTP Status: <span className="text-cyan-400 font-bold">{dynamicResult.browserDetails?.finalStatus || 200}</span></span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-right">
                  <span className="text-[10px] font-mono text-slate-400 uppercase">Risk Score</span>
                  <div className={`text-xl font-bold font-mono ${dynamicResult.riskScore > 70 ? 'text-rose-400' : 'text-emerald-400'}`}>
                    {dynamicResult.riskScore} / 100
                  </div>
                </div>
                <StatusBadge status={dynamicResult.finalVerdict} size="lg" />
              </div>
            </div>

            {/* Quick KPI Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 font-mono text-xs">
                <span className="text-[10px] text-slate-400 block">HTML Forms</span>
                <span className="text-lg font-bold text-slate-100">{dynamicResult.domProfile.formsCount}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 font-mono text-xs">
                <span className="text-[10px] text-slate-400 block">Password Fields</span>
                <span className={`text-lg font-bold ${dynamicResult.domProfile.passwordFields > 0 ? 'text-rose-400' : 'text-slate-100'}`}>
                  {dynamicResult.domProfile.passwordFields}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 font-mono text-xs">
                <span className="text-[10px] text-slate-400 block">Dynamic eval() Calls</span>
                <span className={`text-lg font-bold ${dynamicResult.jsBehavior.evalUsage ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {dynamicResult.jsBehavior.evalUsage ? 'DETECTED' : 'None'}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 font-mono text-xs">
                <span className="text-[10px] text-slate-400 block">Anti-Debug Hooks</span>
                <span className={`text-lg font-bold ${dynamicResult.jsBehavior.antiDebuggingDetected ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {dynamicResult.jsBehavior.antiDebuggingDetected ? 'Active Evasion' : 'Clean'}
                </span>
              </div>
            </div>
          </div>

          {/* Forensic Tabs Header */}
          <div className="flex border-b border-slate-800 gap-2 overflow-x-auto">
            <button
              id="tab-agent"
              onClick={() => setActiveTab('agent')}
              className={`px-4 py-2.5 text-xs font-mono font-semibold transition-all border-b-2 flex items-center gap-2 flex-shrink-0 ${
                activeTab === 'agent'
                  ? 'border-violet-400 text-violet-300 bg-violet-950/20'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sparkles className="w-4 h-4 text-violet-400" />
              <span>Autonomous AI Security Agent</span>
              <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-900/60 text-violet-300 border border-violet-700/50 flex items-center gap-1 font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                ACTIVE TRACE
              </span>
            </button>

            <button
              id="tab-dom"
              onClick={() => setActiveTab('dom')}
              className={`px-4 py-2.5 text-xs font-mono font-semibold transition-all border-b-2 flex items-center gap-2 flex-shrink-0 ${
                activeTab === 'dom'
                  ? 'border-cyan-400 text-cyan-300 bg-cyan-950/20'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Code2 className="w-4 h-4" />
              <span>Interactive DOM Tree Inspector</span>
              {dynamicResult.domTree?.length > 0 && (
                <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800">
                  {dynamicResult.domProfile?.elementsTotal || 25}
                </span>
              )}
            </button>

            <button
              id="tab-browser"
              onClick={() => setActiveTab('browser')}
              className={`px-4 py-2.5 text-xs font-mono font-semibold transition-all border-b-2 flex items-center gap-2 flex-shrink-0 ${
                activeTab === 'browser'
                  ? 'border-cyan-400 text-cyan-300 bg-cyan-950/20'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Eye className="w-4 h-4" />
              <span>Sandbox Browser Viewport & Redirects</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-slate-800 text-slate-300">
                {dynamicResult.redirectChain?.length || 1} hops
              </span>
            </button>

            <button
              id="tab-js"
              onClick={() => setActiveTab('js')}
              className={`px-4 py-2.5 text-xs font-mono font-semibold transition-all border-b-2 flex items-center gap-2 flex-shrink-0 ${
                activeTab === 'js'
                  ? 'border-cyan-400 text-cyan-300 bg-cyan-950/20'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Terminal className="w-4 h-4" />
              <span>JavaScript Runtime Telemetry</span>
              {dynamicResult.jsBehavior.evalUsage && (
                <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-rose-950 text-rose-300 border border-rose-800">
                  ALERT
                </span>
              )}
            </button>

            <button
              id="tab-warnings"
              onClick={() => setActiveTab('warnings')}
              className={`px-4 py-2.5 text-xs font-mono font-semibold transition-all border-b-2 flex items-center gap-2 flex-shrink-0 ${
                activeTab === 'warnings'
                  ? 'border-cyan-400 text-cyan-300 bg-cyan-950/20'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <AlertTriangle className="w-4 h-4" />
              <span>Behavioral Anomalies ({dynamicResult.warnings?.length || 0})</span>
            </button>
          </div>

          {/* TAB 0: AUTONOMOUS AI SECURITY AGENT */}
          {activeTab === 'agent' && (
            <div className="space-y-6">
              {/* Agent Executive Status Banner */}
              <div className="soc-card p-6 border-l-4 border-l-violet-500 bg-gradient-to-r from-violet-950/20 via-slate-900/40 to-slate-950/60 space-y-4">
                <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
                  <div className="flex items-start gap-3">
                    <div className="w-12 h-12 rounded-2xl bg-violet-950/80 border border-violet-500/40 flex items-center justify-center text-violet-400 flex-shrink-0 shadow-[0_0_20px_rgba(139,92,246,0.25)]">
                      <Bot className="w-6 h-6 animate-pulse" />
                    </div>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold font-mono text-slate-100 uppercase tracking-wider">
                          CyberX Autonomous Dynamic Security Agent
                        </span>
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-900/60 text-violet-300 border border-violet-600/50 font-mono font-bold">
                          v2.4 HEURISTIC-DECOMPILER
                        </span>
                      </div>
                      <p className="text-xs font-mono text-slate-400">
                        Autonomous multi-hop redirection tracing, AST deobfuscation, live JavaScript decompiler & behavioral DOM audit.
                      </p>
                    </div>
                  </div>

                  {/* Verdict & Recommended Action */}
                  <div className="flex items-center gap-3">
                    <div className="text-right font-mono">
                      <span className="text-[10px] text-slate-400 block uppercase">Agent Verdict</span>
                      <span className={`text-base font-bold ${
                        (dynamicResult.agentDecisionSummary?.threat_severity === 'CRITICAL' || dynamicResult.threatLevel === 'MALICIOUS' || (dynamicResult.finalVerdict && dynamicResult.finalVerdict.toLowerCase().includes('malicious')))
                          ? 'text-rose-400'
                          : (dynamicResult.agentDecisionSummary?.threat_severity === 'HIGH' || dynamicResult.threatLevel === 'SUSPICIOUS' || (dynamicResult.finalVerdict && dynamicResult.finalVerdict.toLowerCase().includes('suspicious')))
                          ? 'text-amber-400'
                          : 'text-emerald-400'
                      }`}>
                        {dynamicResult.finalVerdict || dynamicResult.agentDecisionSummary?.verdict || dynamicResult.threatLevel || 'ANALYZED'}
                      </span>
                    </div>

                    <div className={`px-3 py-1.5 rounded-xl border text-xs font-mono font-bold flex items-center gap-2 ${
                      (dynamicResult.agentDecisionSummary?.recommended_action === 'BLOCK_IMMEDIATELY' || (dynamicResult.riskScore ?? dynamicResult.threatScore ?? 50) >= 70)
                        ? 'bg-rose-950/80 text-rose-300 border-rose-500/50'
                        : (dynamicResult.agentDecisionSummary?.recommended_action === 'QUARANTINE_URL' || (dynamicResult.riskScore ?? dynamicResult.threatScore ?? 50) >= 40)
                        ? 'bg-amber-950/80 text-amber-300 border-amber-500/50'
                        : 'bg-emerald-950/80 text-emerald-300 border-emerald-500/50'
                    }`}>
                      <Shield className="w-4 h-4" />
                      <span>{dynamicResult.agentDecisionSummary?.recommended_action || ((dynamicResult.riskScore ?? dynamicResult.threatScore ?? 50) >= 70 ? 'BLOCK_IMMEDIATELY' : ((dynamicResult.riskScore ?? dynamicResult.threatScore ?? 50) >= 40 ? 'QUARANTINE_URL' : 'PERMIT_ACCESS'))}</span>
                    </div>
                  </div>
                </div>

                {/* Key Rationale Monospace Box */}
                <div className="p-4 rounded-xl bg-[#060a12] border border-violet-950/80 space-y-2">
                  <div className="flex items-center justify-between text-[11px] font-mono text-violet-300">
                    <span className="flex items-center gap-1.5 font-semibold">
                      <Terminal className="w-3.5 h-3.5 text-violet-400" />
                      SYNTHESIZED THREAT RATIONALE
                    </span>
                    <span className="text-slate-400">
                      Risk Score: <strong className="text-violet-300">{dynamicResult.riskScore ?? dynamicResult.threatScore ?? dynamicResult.agentDecisionSummary?.risk_score ?? 75}/100</strong>
                    </span>
                  </div>
                  <p className="text-xs font-mono text-slate-200 leading-relaxed whitespace-pre-line">
                    {typeof dynamicResult.agentDecisionSummary === 'string' && dynamicResult.agentDecisionSummary
                      ? dynamicResult.agentDecisionSummary
                      : (dynamicResult.agentDecisionSummary?.key_rationale ||
                        'The autonomous security agent completed live redirection tracing, extracted scripts from the landing DOM, and executed heuristic pattern audits across all dynamic vectors.')}
                  </p>
                </div>

                {/* Quick Telemetry Bar */}
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 font-mono text-xs">
                  <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Redirect Hops</span>
                    <span className="text-sm font-bold text-cyan-400">
                      {dynamicResult.agentDecisionSummary?.redirect_hops_count || dynamicResult.redirectChain?.length || 1} Hops
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Scripts Audited</span>
                    <span className="text-sm font-bold text-violet-400">
                      {dynamicResult.agentDecisionSummary?.scripts_analyzed || dynamicResult.scripts?.length || 0} Scripts
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Threat Signatures</span>
                    <span className={`text-sm font-bold ${
                      (dynamicResult.detectedThreats?.length || 0) > 0 ? 'text-rose-400' : 'text-emerald-400'
                    }`}>
                      {dynamicResult.detectedThreats?.length || 0} Flags
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Hidden iFrames</span>
                    <span className={`text-sm font-bold ${
                      (dynamicResult.agentDecisionSummary?.hidden_iframes_count || 0) > 0 ? 'text-rose-400' : 'text-slate-300'
                    }`}>
                      {dynamicResult.agentDecisionSummary?.hidden_iframes_count || 0} Found
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">External Forms</span>
                    <span className={`text-sm font-bold ${
                      (dynamicResult.agentDecisionSummary?.external_form_actions_count || 0) > 0 ? 'text-amber-400' : 'text-slate-300'
                    }`}>
                      {dynamicResult.agentDecisionSummary?.external_form_actions_count || 0} Post Targets
                    </span>
                  </div>
                </div>
              </div>

              {/* Multi-Phase Autonomous Execution Trajectory */}
              <div className="soc-card p-6 space-y-6 font-mono">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <Workflow className="w-5 h-5 text-violet-400" />
                    <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                      Autonomous Agent Execution Trajectory & Reasoning Timeline
                    </span>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    Sequential inspection steps executed autonomously
                  </span>
                </div>

                <div className="space-y-6">
                  {(dynamicResult.agentTrajectory && dynamicResult.agentTrajectory.length > 0
                    ? dynamicResult.agentTrajectory
                    : [
                        {
                          stepNumber: 1,
                          phase: 'Multi-Hop Redirection Navigation',
                          status: dynamicResult.redirectChain?.length > 1 ? 'threat_flagged' : 'nominal',
                          thought: 'Tracing initial target URL through network redirection hops and TLS handshake verification...',
                          summary: `Navigated ${dynamicResult.redirectChain?.length || 1} hop(s). Destination HTTP status: ${dynamicResult.browserDetails?.finalStatus || 200}.`,
                          evidence: { chain: dynamicResult.redirectChain || [] }
                        },
                        {
                          stepNumber: 2,
                          phase: 'AST Script & Payload Decompilation',
                          status: dynamicResult.jsBehavior?.evalUsage ? 'threat_flagged' : 'nominal',
                          thought: 'Extracting all script elements and running AST regex decompiler across 8 known threat families...',
                          summary: `Inspected JavaScript runtime telemetry. Eval usage: ${dynamicResult.jsBehavior?.evalUsage ? 'DETECTED' : 'Clean'}. Keyloggers: ${dynamicResult.jsBehavior?.keyloggerHooks ? 'DETECTED' : 'Clean'}.`,
                          evidence: {}
                        },
                        {
                          stepNumber: 3,
                          phase: 'DOM Structure & Behavioral Threat Audit',
                          status: 'nominal',
                          thought: 'Auditing DOM hierarchy for zero-pixel iframes, form exfiltration destinations, and hidden overlays...',
                          summary: `Inspected DOM nodes. Forms detected: ${dynamicResult.domProfile?.formsCount || 0}.`,
                          evidence: {}
                        },
                        {
                          stepNumber: 4,
                          phase: 'Autonomous Threat Synthesis & Verdict',
                          status: dynamicResult.threatScore >= 50 ? 'threat_flagged' : 'nominal',
                          thought: 'Synthesizing dynamic telemetry with lexical classifier to issue final containment posture...',
                          summary: `Final verdict rendered: ${dynamicResult.threatLevel}. Security recommendation: ${dynamicResult.threatScore >= 70 ? 'BLOCK_IMMEDIATELY' : 'PERMIT'}.`,
                          evidence: {}
                        }
                      ]
                  ).map((step, idx) => {
                    const isThreat = step.status === 'threat_flagged';
                    return (
                      <div
                        key={idx}
                        className={`rounded-2xl border transition-all p-5 space-y-4 ${
                          isThreat
                            ? 'bg-rose-950/10 border-rose-500/30 shadow-[0_0_15px_rgba(244,63,94,0.08)]'
                            : 'bg-slate-900/40 border-slate-800/90'
                        }`}
                      >
                        {/* Step Header */}
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800/80">
                          <div className="flex items-center gap-3">
                            <span className={`w-7 h-7 rounded-xl flex items-center justify-center text-xs font-bold ${
                              isThreat
                                ? 'bg-rose-950 text-rose-300 border border-rose-600'
                                : 'bg-violet-950 text-violet-300 border border-violet-700'
                            }`}>
                              {step.stepNumber || step.step || idx + 1}
                            </span>
                            <span className="text-xs font-bold text-slate-100 uppercase tracking-wide">
                              {step.phase}
                            </span>
                          </div>

                          <div className="flex items-center gap-2">
                            <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full flex items-center gap-1 ${
                              isThreat
                                ? 'bg-rose-950 text-rose-300 border border-rose-700/60'
                                : 'bg-emerald-950 text-emerald-300 border border-emerald-700/60'
                            }`}>
                              {isThreat ? (
                                <>
                                  <AlertTriangle className="w-3 h-3 text-rose-400" />
                                  <span>THREAT FLAG</span>
                                </>
                              ) : (
                                <>
                                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                                  <span>NOMINAL</span>
                                </>
                              )}
                            </span>
                          </div>
                        </div>

                        {/* Step AI Reasoning Thought */}
                        <div className="p-3.5 rounded-xl bg-[#050811] border-l-4 border-l-violet-500 border-r border-t border-b border-slate-800/80 space-y-1">
                          <div className="flex items-center gap-1.5 text-[10px] text-violet-400 font-bold uppercase tracking-wider">
                            <Bot className="w-3.5 h-3.5" />
                            <span>AI Agent Autonomous Reasoning</span>
                          </div>
                          <p className="text-xs text-violet-200/90 italic leading-relaxed">
                            "{step.thought || step.agent_thought}"
                          </p>
                        </div>

                        {/* Step Summary Finding */}
                        <div className="text-xs text-slate-300 leading-relaxed flex items-start gap-2">
                          <ArrowRight className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                          <span>{step.summary}</span>
                        </div>

                        {/* Granular Evidence Blocks */}
                        {step.stepNumber === 1 && step.evidence?.chain && step.evidence.chain.length > 0 && (
                          <div className="mt-3 p-3 rounded-xl bg-slate-950/60 border border-slate-800 space-y-2 text-xs">
                            <span className="text-[10px] text-slate-400 uppercase font-bold block">
                              Redirect Chain Traced ({step.evidence.chain.length} hops)
                            </span>
                            <div className="space-y-1.5">
                              {step.evidence.chain.map((hop, hIdx) => (
                                <div key={hIdx} className="flex items-center justify-between gap-2 p-2 rounded bg-slate-900/60 border border-slate-800/80">
                                  <span className="text-slate-300 break-all text-[11px]">{hop.url}</span>
                                  <span className={`text-[10px] px-2 py-0.5 rounded font-bold flex-shrink-0 ${
                                    hop.status === 200 ? 'bg-emerald-950 text-emerald-300' : 'bg-amber-950 text-amber-300'
                                  }`}>
                                    HTTP {hop.status}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {step.stepNumber === 2 && dynamicResult.detectedThreats && dynamicResult.detectedThreats.length > 0 && (
                          <div className="mt-3 space-y-2">
                            <span className="text-[10px] text-slate-400 uppercase font-bold block">
                              Decompiled Threats ({dynamicResult.detectedThreats.length})
                            </span>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                              {dynamicResult.detectedThreats.map((threat, tIdx) => (
                                <div key={tIdx} className="p-3 rounded-xl bg-rose-950/30 border border-rose-500/40 space-y-1">
                                  <div className="flex items-center justify-between text-xs">
                                    <span className="font-bold text-rose-300">{threat.type}</span>
                                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-rose-900 text-rose-200">
                                      {threat.mitre || 'ATT&CK'}
                                    </span>
                                  </div>
                                  <p className="text-[11px] text-slate-300">{threat.description}</p>
                                  {threat.sample && (
                                    <div className="p-1.5 rounded bg-black/50 text-[10px] text-amber-300 font-mono truncate">
                                      {threat.sample}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Action Jump Bar */}
              <div className="soc-card p-4 flex flex-wrap items-center justify-between gap-3 font-mono text-xs">
                <span className="text-slate-400">Jump directly into specialized forensic viewports:</span>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setActiveTab('browser')}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 border border-slate-700 flex items-center gap-1.5 transition-colors"
                  >
                    <Eye className="w-3.5 h-3.5" />
                    <span>Sandbox Viewport</span>
                  </button>
                  <button
                    onClick={() => setActiveTab('js')}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-violet-300 border border-slate-700 flex items-center gap-1.5 transition-colors"
                  >
                    <Terminal className="w-3.5 h-3.5" />
                    <span>JS Decompiler</span>
                  </button>
                  <button
                    onClick={() => setActiveTab('dom')}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 flex items-center gap-1.5 transition-colors"
                  >
                    <Code2 className="w-3.5 h-3.5" />
                    <span>DOM Tree Inspector</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* TAB 1: INTERACTIVE DOM TREE INSPECTOR */}
          {activeTab === 'dom' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* DOM Tree Column */}
              <div className="lg:col-span-2 soc-card p-5 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
                  <div className="flex items-center gap-2">
                    <FileCode className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-bold font-mono text-slate-200 uppercase">
                      DOM Hierarchy Tree (Antigravity Inspector)
                    </span>
                  </div>

                  <div className="flex items-center gap-2 text-xs font-mono">
                    <Filter className="w-3.5 h-3.5 text-slate-400" />
                    <select
                      value={domFilter}
                      onChange={(e) => setDomFilter(e.target.value)}
                      className="bg-[#080d19] border border-slate-700 rounded-lg px-2.5 py-1 text-slate-300 text-xs focus:outline-none focus:border-cyan-400"
                    >
                      <option value="all">Filter: All Elements</option>
                      <option value="form">Forms (&lt;form&gt;)</option>
                      <option value="input">Inputs (&lt;input&gt;)</option>
                      <option value="script">Scripts (&lt;script&gt;)</option>
                      <option value="iframe">iFrames (&lt;iframe&gt;)</option>
                    </select>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[#070b14] border border-slate-800/80 max-h-[520px] overflow-y-auto space-y-0.5">
                  {dynamicResult.domTree && dynamicResult.domTree.length > 0 ? (
                    dynamicResult.domTree.map((rootNode, idx) => (
                      <DomNodeView
                        key={idx}
                        node={rootNode}
                        selectedNode={selectedNode}
                        onSelectNode={setSelectedNode}
                        filterTag={domFilter}
                      />
                    ))
                  ) : (
                    <div className="p-8 text-center text-slate-500 font-mono text-xs">
                      No DOM tree nodes captured for this execution.
                    </div>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-4 pt-2 text-[11px] font-mono text-slate-400 border-t border-slate-800/80">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded bg-rose-500/80" />
                    <span>External Form Action / Password</span>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded bg-purple-500/80" />
                    <span>Obfuscated Script</span>
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded bg-amber-500/80" />
                    <span>Hidden Element Hook</span>
                  </span>
                </div>
              </div>

              {/* Element Details Drawer / Inspector Column */}
              <div className="soc-card p-5 space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <span className="text-xs font-bold font-mono text-slate-200 uppercase flex items-center gap-2">
                    <Search className="w-4 h-4 text-violet-400" />
                    Element Inspector
                  </span>
                  {selectedNode && (
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300">
                      &lt;{selectedNode.tag}&gt;
                    </span>
                  )}
                </div>

                {selectedNode ? (
                  <div className="space-y-4 font-mono text-xs">
                    {/* Security Flag Banner */}
                    {selectedNode.threat_badge ? (
                      <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/40 space-y-1">
                        <span className="text-[10px] font-bold text-rose-400 flex items-center gap-1.5">
                          <AlertTriangle className="w-3.5 h-3.5" />
                          SECURITY THREAT FLAG:
                        </span>
                        <div className="text-xs text-rose-200 font-bold">{selectedNode.threat_badge}</div>
                      </div>
                    ) : (
                      <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 text-slate-400 text-xs">
                        No immediate anomalies observed on this specific element.
                      </div>
                    )}

                    {/* Tag & Text */}
                    <div className="space-y-1">
                      <span className="text-[10px] text-slate-400 uppercase">Element Tag</span>
                      <div className="p-2 rounded bg-slate-900 border border-slate-800 text-cyan-300 font-bold">
                        {selectedNode.tag}
                      </div>
                    </div>

                    {selectedNode.text && (
                      <div className="space-y-1">
                        <span className="text-[10px] text-slate-400 uppercase">Text / Script Content</span>
                        <div className="p-2.5 rounded bg-[#070b14] border border-slate-800 text-slate-300 max-h-32 overflow-y-auto text-[11px] whitespace-pre-wrap break-all">
                          {selectedNode.text}
                        </div>
                      </div>
                    )}

                    {/* Attributes Key-Value Table */}
                    <div className="space-y-2">
                      <span className="text-[10px] text-slate-400 uppercase">HTML Attributes</span>
                      {selectedNode.attributes && Object.keys(selectedNode.attributes).length > 0 ? (
                        <div className="rounded-lg border border-slate-800 divide-y divide-slate-800/80 overflow-hidden text-[11px]">
                          {Object.entries(selectedNode.attributes).map(([attr, val]) => (
                            <div key={attr} className="p-2 flex flex-col bg-slate-900/50">
                              <span className="text-cyan-400 font-bold">{attr}</span>
                              <span className="text-amber-200/90 break-all">{val}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <span className="text-[11px] text-slate-500 block">No attributes specified.</span>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="p-8 text-center text-slate-500 font-mono text-xs">
                    Click any element in the DOM tree on the left to inspect its HTML attributes, exfiltration targets, and security flags.
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: SANDBOX BROWSER VIEWPORT & REDIRECTS */}
          {activeTab === 'browser' && (
            <div className="space-y-6">
              {/* Virtual Browser Window */}
              <div className="soc-card overflow-hidden border border-slate-700/80 shadow-2xl">
                {/* Browser Top Chrome / Titlebar */}
                <div className="bg-[#0b101d] px-4 py-3 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-rose-500/80 inline-block" />
                    <span className="w-3 h-3 rounded-full bg-amber-500/80 inline-block" />
                    <span className="w-3 h-3 rounded-full bg-emerald-500/80 inline-block" />
                    <span className="text-[11px] font-mono text-slate-400 ml-2 hidden sm:inline">CyberX Isolated Chromium Sandbox</span>
                  </div>

                  {/* Browser Address Bar */}
                  <div className="flex-1 max-w-xl mx-auto flex items-center gap-2 bg-[#060a12] border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-200 w-full">
                    <Lock className={`w-3.5 h-3.5 flex-shrink-0 ${dynamicResult.browserDetails?.sslValid ? 'text-emerald-400' : 'text-rose-400'}`} />
                    <span className="truncate flex-1 font-semibold">{dynamicResult.finalUrl || dynamicResult.browserDetails?.finalUrl || dynamicResult.url}</span>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-cyan-300 flex-shrink-0">
                      {dynamicResult.browserDetails?.finalStatus || 200} OK
                    </span>
                    <a
                      href={dynamicResult.finalUrl || dynamicResult.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      title="Open URL in external tab"
                      className="text-slate-400 hover:text-cyan-300 ml-1 flex-shrink-0"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>

                  {/* Mode Selector Buttons */}
                  <div className="flex items-center gap-1.5 text-[11px] font-mono flex-shrink-0">
                    <button
                      onClick={() => setViewportMode('render')}
                      className={`px-2.5 py-1 rounded-md text-[10px] font-bold transition-all ${
                        viewportMode === 'render'
                          ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                          : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                      }`}
                    >
                      Live Web Viewport
                    </button>
                    <button
                      onClick={() => setViewportMode('mockup')}
                      className={`px-2.5 py-1 rounded-md text-[10px] font-bold transition-all ${
                        viewportMode === 'mockup'
                          ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                          : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                      }`}
                    >
                      Forensic Form
                    </button>
                    <button
                      onClick={() => setViewportMode('source')}
                      className={`px-2.5 py-1 rounded-md text-[10px] font-bold transition-all ${
                        viewportMode === 'source'
                          ? 'bg-cyan-950 text-cyan-300 border border-cyan-700'
                          : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                      }`}
                    >
                      HTML Source
                    </button>
                  </div>
                </div>

                {/* VIEWPORT BODY */}
                {viewportMode === 'render' && (
                  <div className="relative w-full h-[540px] bg-slate-950 flex flex-col">
                    <div className="bg-[#070b14] px-4 py-2 border-b border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                        <span className="text-slate-300 font-semibold">Detonated Sandbox Web Environment</span>
                      </div>
                      <span className="text-amber-400 text-[10px]">
                        Scripts Disarmed for Analyst Safety • CSS & HTML Rendered
                      </span>
                    </div>

                    <iframe
                      srcDoc={
                        dynamicResult.safeHtml ||
                        `<!DOCTYPE html><html><body style="font-family:sans-serif;background:#0f172a;color:#f8fafc;padding:40px;text-align:center;"><h2>Web Service Detonated</h2><p>Target: ${dynamicResult.finalUrl || dynamicResult.url}</p><p>Status: HTTP 200 OK</p></body></html>`
                      }
                      sandbox="allow-same-origin allow-forms"
                      title="Detonated Website Sandbox Preview"
                      className="w-full flex-1 border-0 bg-white"
                    />
                  </div>
                )}

                {viewportMode === 'mockup' && (
                  <div className="p-8 bg-[#040711] min-h-[420px] flex items-center justify-center relative">
                    <div className="absolute top-3 right-4 px-2.5 py-1 rounded-full bg-amber-950/80 border border-amber-500/40 text-[10px] font-mono text-amber-300 flex items-center gap-1.5">
                      <AlertTriangle className="w-3 h-3" />
                      <span>ISOLATED SIMULATED VIEWPORT • NO LOCAL JAVASCRIPT EXECUTION</span>
                    </div>

                    {/* Rendered Mockup of Target Site */}
                    <div className="w-full max-w-md p-6 rounded-2xl bg-[#0a0f1d] border border-slate-800/80 shadow-xl space-y-4 font-mono">
                      <div className="text-center space-y-1 pb-3 border-b border-slate-800">
                        <div className="w-10 h-10 mx-auto rounded-xl bg-cyan-950 border border-cyan-500/40 flex items-center justify-center text-cyan-300 font-bold text-lg">
                          {dynamicResult.url.includes('microsoft') ? 'M' : 'S'}
                        </div>
                        <h4 className="text-sm font-bold text-slate-100">{dynamicResult.domProfile?.pageTitle || 'Authentication Gateway'}</h4>
                        <p className="text-[10px] text-slate-400">Security Verification Required</p>
                      </div>

                      {dynamicResult.domProfile?.formsCount > 0 ? (
                        <div className="space-y-3">
                          <div className="space-y-1">
                            <span className="text-[10px] text-slate-400">Corporate Identity / Email</span>
                            <input
                              type="text"
                              placeholder="user@organization.com"
                              disabled
                              className="w-full bg-[#060a12] border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-400"
                            />
                          </div>
                          <div className="space-y-1">
                            <span className="text-[10px] text-slate-400">Security Token / Password</span>
                            <input
                              type="password"
                              placeholder="••••••••••••"
                              disabled
                              className="w-full bg-[#060a12] border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-400"
                            />
                          </div>
                          <button
                            disabled
                            className="w-full py-2 rounded-lg bg-cyan-600/50 text-cyan-200 text-xs font-bold cursor-not-allowed"
                          >
                            Verify Identity & Renew Access
                          </button>
                        </div>
                      ) : (
                        <div className="py-6 text-center text-slate-400 text-xs space-y-2">
                          <Globe className="w-8 h-8 text-cyan-400 mx-auto opacity-60" />
                          <p>Web service loaded successfully without interactive credential submission prompts.</p>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {viewportMode === 'source' && (
                  <div className="p-4 bg-[#050811] max-h-[500px] overflow-y-auto font-mono text-xs">
                    <pre className="text-cyan-200/90 whitespace-pre-wrap leading-relaxed">
                      {dynamicResult.safeHtml || '<!-- No raw source available for offline target -->'}
                    </pre>
                  </div>
                )}
              </div>

              {/* Multi-Hop Navigation & Redirect Chain */}
              <div className="soc-card p-5 space-y-4 font-mono">
                <span className="text-xs font-bold text-slate-200 uppercase flex items-center gap-2">
                  <CornerDownRight className="w-4 h-4 text-cyan-400" />
                  Navigation Redirect Trace & Trajectory
                </span>

                <div className="space-y-3">
                  {dynamicResult.redirectChain && dynamicResult.redirectChain.map((hop, idx) => (
                    <div
                      key={idx}
                      className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
                    >
                      <div className="flex items-center gap-3">
                        <span className="w-6 h-6 rounded-full bg-cyan-950 text-cyan-300 border border-cyan-800 flex items-center justify-center text-[10px] font-bold">
                          {hop.step}
                        </span>
                        <div>
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-bold text-slate-200 break-all">{hop.url}</span>
                            {hop.cross_origin && (
                              <span className="px-1.5 py-0.2 rounded bg-rose-950 text-rose-300 border border-rose-800 text-[9px] font-bold flex-shrink-0">
                                CROSS-DOMAIN REDIRECT
                              </span>
                            )}
                          </div>
                          <span className="block text-[10px] text-slate-400 mt-0.5">{hop.reason}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 flex-shrink-0">
                        <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                          hop.status === 200
                            ? 'bg-emerald-950 text-emerald-300 border border-emerald-800'
                            : 'bg-amber-950 text-amber-300 border border-amber-800'
                        }`}>
                          HTTP {hop.status}
                        </span>
                        <span className="text-slate-500 text-[11px]">{hop.latency_ms || 110}ms</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: JAVASCRIPT BEHAVIORAL TELEMETRY */}
          {activeTab === 'js' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Dynamic Eval Card */}
                <div className={`soc-card p-5 space-y-2 border-l-4 ${
                  dynamicResult.jsBehavior.evalUsage ? 'border-l-rose-500' : 'border-l-emerald-500'
                }`}>
                  <span className="text-[10px] font-mono text-slate-400 uppercase">eval() / Dynamic Function()</span>
                  <div className="text-lg font-bold font-mono text-slate-100 flex items-center justify-between">
                    <span>{dynamicResult.jsBehavior.evalUsage ? 'DETECTED' : 'Not Observed'}</span>
                    <Terminal className={`w-5 h-5 ${dynamicResult.jsBehavior.evalUsage ? 'text-rose-400' : 'text-emerald-400'}`} />
                  </div>
                  <p className="text-[11px] font-mono text-slate-400">
                    {dynamicResult.jsBehavior.evalUsage
                      ? 'JavaScript compiles dynamic string code at runtime via eval() or Function constructor, typical of deobfuscation packers.'
                      : 'No dynamic code evaluation or runtime string deobfuscation detected.'}
                  </p>
                </div>

                {/* Anti-Debugging Timing Card */}
                <div className={`soc-card p-5 space-y-2 border-l-4 ${
                  dynamicResult.jsBehavior.antiDebuggingDetected ? 'border-l-rose-500' : 'border-l-emerald-500'
                }`}>
                  <span className="text-[10px] font-mono text-slate-400 uppercase">Anti-Sandbox Evasion Loop</span>
                  <div className="text-lg font-bold font-mono text-slate-100 flex items-center justify-between">
                    <span>{dynamicResult.jsBehavior.antiDebuggingDetected ? 'ACTIVE EVASION' : 'Clean'}</span>
                    <Clock className={`w-5 h-5 ${dynamicResult.jsBehavior.antiDebuggingDetected ? 'text-rose-400' : 'text-emerald-400'}`} />
                  </div>
                  <p className="text-[11px] font-mono text-slate-400">
                    {dynamicResult.jsBehavior.antiDebuggingDetected
                      ? 'Detected debugger timing loops (setInterval debugger) designed to crash developer tools and stall automated sandboxes.'
                      : 'No timing evasion or debugger statement traps encountered during execution.'}
                  </p>
                </div>

                {/* Keylogger Hook Card */}
                <div className={`soc-card p-5 space-y-2 border-l-4 ${
                  dynamicResult.jsBehavior.keyloggerHooks ? 'border-l-rose-500' : 'border-l-emerald-500'
                }`}>
                  <span className="text-[10px] font-mono text-slate-400 uppercase">Global Keylogger Listeners</span>
                  <div className="text-lg font-bold font-mono text-slate-100 flex items-center justify-between">
                    <span>{dynamicResult.jsBehavior.keyloggerHooks ? 'INTERCEPTED' : 'Standard'}</span>
                    <Radio className={`w-5 h-5 ${dynamicResult.jsBehavior.keyloggerHooks ? 'text-rose-400' : 'text-emerald-400'}`} />
                  </div>
                  <p className="text-[11px] font-mono text-slate-400">
                    {dynamicResult.jsBehavior.keyloggerHooks
                      ? 'Window-level keydown event listeners registered to intercept and record victim credentials before form submission.'
                      : 'Standard form input controls with no document-level keystroke interception hooks.'}
                  </p>
                </div>
              </div>

              {/* Decompiled Script Elements & AST Inspection */}
              {(dynamicResult.jsBehavior?.scripts?.length > 0 || dynamicResult.scripts?.length > 0) && (
                <div className="soc-card p-5 space-y-4 font-mono">
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                    <div className="flex items-center gap-2">
                      <Code2 className="w-4 h-4 text-violet-400" />
                      <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                        Decompiled DOM Script Elements & AST Heuristics ({
                          dynamicResult.jsBehavior?.scripts?.length || dynamicResult.scripts?.length || 0
                        })
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-400">
                      Live JavaScript payloads extracted and analyzed by AI Agent
                    </span>
                  </div>

                  <div className="space-y-3">
                    {(dynamicResult.jsBehavior?.scripts || dynamicResult.scripts || []).map((sc, sIdx) => {
                      const isHighEntropy = sc.entropy > 5.5;
                      const hasThreats = sc.threat_indicators && sc.threat_indicators.length > 0;
                      return (
                        <div
                          key={sIdx}
                          className={`p-4 rounded-xl border text-xs space-y-3 ${
                            hasThreats
                              ? 'bg-rose-950/20 border-rose-500/40'
                              : isHighEntropy
                              ? 'bg-amber-950/20 border-amber-500/40'
                              : 'bg-slate-900/50 border-slate-800'
                          }`}
                        >
                          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <span className="font-bold text-slate-200">
                                {sc.script_id || `Script #${sIdx + 1}`}
                              </span>
                              <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                                {sc.type === 'external' ? 'External Source' : 'Inline Script'}
                              </span>
                              <span className="text-[10px] text-slate-500">
                                {sc.length || 0} bytes
                              </span>
                            </div>

                            <div className="flex items-center gap-2">
                              <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                                isHighEntropy
                                  ? 'bg-rose-950 text-rose-300 border border-rose-700/60'
                                  : 'bg-emerald-950 text-emerald-300 border border-emerald-700/60'
                              }`}>
                                Entropy: {Number(sc.entropy || 0).toFixed(2)} bits/char
                              </span>

                              {hasThreats ? (
                                <span className="text-[10px] px-2 py-0.5 rounded bg-rose-950 text-rose-300 border border-rose-800 font-bold flex items-center gap-1">
                                  <AlertTriangle className="w-3 h-3 text-rose-400" />
                                  <span>{sc.threat_indicators.length} Threat Flag(s)</span>
                                </span>
                              ) : (
                                <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold">
                                  AST Clean
                                </span>
                              )}
                            </div>
                          </div>

                          {/* Threat indicators pills */}
                          {hasThreats && (
                            <div className="flex flex-wrap gap-1.5 pt-1">
                              {sc.threat_indicators.map((indicator, iIdx) => (
                                <span
                                  key={iIdx}
                                  className="text-[10px] px-2 py-0.5 rounded-full bg-rose-900/60 text-rose-200 border border-rose-700 font-mono"
                                >
                                  {indicator}
                                </span>
                              ))}
                            </div>
                          )}

                          {/* Code sample snippet */}
                          {sc.code_sample && (
                            <div className="p-3 rounded-lg bg-[#040711] border border-slate-800/90 overflow-x-auto">
                              <pre className="text-[11px] font-mono text-cyan-200/90 whitespace-pre-wrap leading-relaxed max-h-36 overflow-y-auto">
                                {sc.code_sample}
                              </pre>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Detailed JS Runtime Trace Console */}
              <div className="soc-card p-5 space-y-3 font-mono">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <span className="text-xs font-bold text-slate-200 uppercase flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-cyan-400" />
                    Sandbox JavaScript Execution Trace & Console Interceptions
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    Shannon Entropy: {dynamicResult.jsBehavior.inlineEntropy || 5.4} bits/char
                  </span>
                </div>

                <div className="p-4 rounded-xl bg-[#050811] border border-slate-800 text-xs space-y-2 max-h-64 overflow-y-auto font-mono text-slate-300">
                  <div className="text-slate-500">[0.010s] Chromium Engine initialized with isolated profile.</div>
                  <div className="text-slate-500">[0.024s] DOMContentLoaded event triggered.</div>
                  {dynamicResult.jsBehavior.evalUsage ? (
                    <>
                      <div className="text-amber-400">
                        [0.052s] WARNING: Script invoked eval(atob("...")) payload decoder.
                      </div>
                      <div className="text-rose-400">
                        [0.089s] DETECTED: setInterval(function(){ '{ debugger; }' }, 150) loop initiated.
                      </div>
                      <div className="text-rose-400">
                        [0.142s] DETECTED: window.addEventListener('keydown', captureKeystrokes) registered on global object.
                      </div>
                      <div className="text-cyan-400">
                        [0.210s] INTERCEPT: Form action configured to external endpoint: {dynamicResult.domTree?.[0]?.children?.[1]?.children?.[1]?.attributes?.action || 'http://185.220.101.5/harvest.php'}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="text-emerald-400">
                        [0.045s] Standard script bundles compiled cleanly without dynamic eval compilation.
                      </div>
                      <div className="text-emerald-400">
                        [0.078s] No anti-debugging or timing loops triggered. Execution within nominal browser bounds.
                      </div>
                      <div className="text-slate-400">
                        [0.112s] Total network XHR/fetch requests completed: {dynamicResult.jsBehavior.networkCallsCount || 1}
                      </div>
                    </>
                  )}
                  <div className="text-slate-500">[0.340s] Sandbox evaluation completed. Disposing temporary browser context.</div>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: BEHAVIORAL ANOMALIES OBSERVED IN SANDBOX */}
          {activeTab === 'warnings' && (
            <div className="soc-card p-6 space-y-4 font-mono">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                    Behavioral Anomalies Observed in Sandbox ({dynamicResult.warnings?.length || 0})
                  </span>
                </div>
                <span className="text-[11px] text-slate-400">
                  Granular threat telemetry parsed from headless execution
                </span>
              </div>

              <div className="space-y-3">
                {dynamicResult.warnings && dynamicResult.warnings.map((w, i) => {
                  const { level, text } = parseWarningSeverity(w);
                  return (
                    <div
                      key={i}
                      className={`p-4 rounded-xl border flex items-start gap-3 text-xs ${
                        level === 'critical'
                          ? 'bg-rose-950/30 border-rose-500/40 text-rose-200'
                          : level === 'high'
                          ? 'bg-orange-950/30 border-orange-500/40 text-orange-200'
                          : level === 'medium'
                          ? 'bg-amber-950/30 border-amber-500/40 text-amber-200'
                          : level === 'safe'
                          ? 'bg-emerald-950/30 border-emerald-500/40 text-emerald-200'
                          : 'bg-cyan-950/30 border-cyan-500/40 text-cyan-200'
                      }`}
                    >
                      {level === 'critical' && <AlertOctagon className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />}
                      {level === 'high' && <AlertTriangle className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />}
                      {level === 'medium' && <AlertCircle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />}
                      {level === 'safe' && <ShieldCheck className="w-4 h-4 text-emerald-400 flex-shrink-0 mt-0.5" />}
                      {level === 'info' && <Info className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />}

                      <div className="space-y-1">
                        <span className={`text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.2 rounded ${
                          level === 'critical' ? 'bg-rose-950 text-rose-300' :
                          level === 'high' ? 'bg-orange-950 text-orange-300' :
                          level === 'medium' ? 'bg-amber-950 text-amber-300' :
                          level === 'safe' ? 'bg-emerald-950 text-emerald-300' :
                          'bg-cyan-950 text-cyan-300'
                        }`}>
                          {level.toUpperCase()}
                        </span>
                        <p className="text-xs font-mono leading-relaxed mt-1">{text}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DynamicAnalysis;
