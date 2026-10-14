import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Shield,
  Activity,
  Globe,
  Binary,
  ArrowRight,
  Server,
  Cpu,
  BrainCircuit,
  AlertTriangle,
  CheckCircle2,
  GitFork,
  ExternalLink,
  Clock,
  Sparkles
} from 'lucide-react';
import { ResponsiveContainer, PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis } from 'recharts';
import { useBackend } from '../context/BackendContext';
import { useScans } from '../context/ScanContext';
import MetricCard from '../components/common/MetricCard';
import StatusBadge from '../components/common/StatusBadge';
import EmptyState from '../components/common/EmptyState';
import { formatTimestamp } from '../utils/helpers';

export const Dashboard = () => {
  const navigate = useNavigate();
  const { isOnline, status, latency, apiUrl, moduleStatus } = useBackend();
  const { history, activeScan } = useScans();

  // Compute actual aggregated numbers strictly from real scan history
  const totalScans = history.length;
  const maliciousCount = history.filter(
    (s) =>
      s.riskLevel?.toLowerCase() === 'malicious' ||
      s.riskLevel?.toLowerCase() === 'high' ||
      s.riskLevel?.toLowerCase() === 'critical'
  ).length;
  const suspiciousCount = history.filter(
    (s) => s.riskLevel?.toLowerCase() === 'suspicious' || s.riskLevel?.toLowerCase() === 'medium'
  ).length;
  const benignCount = history.filter(
    (s) => s.riskLevel?.toLowerCase() === 'safe' || s.riskLevel?.toLowerCase() === 'benign'
  ).length;

  // Chart data from actual scan distribution
  const riskDistributionData = [
    { name: 'Malicious', value: maliciousCount, color: '#ef4444' },
    { name: 'Suspicious', value: suspiciousCount, color: '#f59e0b' },
    { name: 'Benign / Safe', value: benignCount, color: '#10b981' },
  ].filter((item) => item.value > 0);

  // Category breakdown from scans
  const categoryCounts = history.reduce((acc, curr) => {
    const cat = curr.classification || 'Uncategorized';
    acc[cat] = (acc[cat] || 0) + 1;
    return acc;
  }, {});

  const categoryChartData = Object.entries(categoryCounts).map(([name, count]) => ({
    name,
    count
  }));

  return (
    <div className="space-y-6">
      {/* Welcome & Project Objective Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-[#0d162b] via-[#091124] to-[#120f26] border border-cyan-500/20 p-6 md:p-8 shadow-glass">
        {/* Subtle grid background */}
        <div className="absolute inset-0 bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:24px_24px] opacity-10 pointer-events-none" />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="max-w-3xl space-y-2.5">
            <div className="flex items-center gap-2">
              <span className="soc-badge-cyan text-[11px] font-mono">EDAI CAPSTONE PROJECT</span>
              <span className="text-xs text-slate-400 font-mono">• Research Framework</span>
            </div>
            <h1 className="text-xl md:text-2xl lg:text-3xl font-extrabold text-slate-100 font-mono tracking-tight">
              A Graph Neural Network-Based Framework for Explainable Zero-Day Malware and Malicious-URL Behavior Analysis
            </h1>
            <p className="text-xs md:text-sm text-slate-300 leading-relaxed max-w-2xl">
              Unified cybersecurity threat intelligence dashboard combining topology-aware graph neural networks,
              volatile memory forensics (CIC-MalMem-2022), static PE classification (BODMAS/EMBER), and transparent SHAP/gradient attribution.
            </p>
          </div>

          <div className="flex flex-wrap lg:flex-col gap-2.5 flex-shrink-0">
            <button
              onClick={() => navigate('/threat-analyzer')}
              className="soc-btn-primary text-xs px-5 py-3 shadow-glow-cyan"
            >
              <Shield className="w-4 h-4" />
              <span>Launch Threat Analyzer</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <div className="flex gap-2">
              <button
                onClick={() => navigate('/url-intelligence')}
                className="soc-btn-secondary text-xs px-3.5 py-2 flex-1"
              >
                <Globe className="w-3.5 h-3.5 text-cyan-400" />
                <span>URL Intel</span>
              </button>
              <button
                onClick={() => navigate('/malware-analysis')}
                className="soc-btn-secondary text-xs px-3.5 py-2 flex-1"
              >
                <Binary className="w-3.5 h-3.5 text-violet-400" />
                <span>Malware</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Real Infrastructure & Operational Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Backend Connectivity"
          value={status === 'checking' ? 'CHECKING' : isOnline ? 'ONLINE' : 'OFFLINE'}
          subtitle={apiUrl}
          icon={Server}
          color={isOnline ? 'emerald' : status === 'checking' ? 'amber' : 'rose'}
          badge={
            isOnline && latency !== null ? (
              <span className="text-[10px] text-emerald-400 font-mono">{latency}ms RTT</span>
            ) : null
          }
        />
        <MetricCard
          title="Total Threat Scans"
          value={totalScans.toString()}
          subtitle="Processed across current session"
          icon={Activity}
          color="cyan"
        />
        <MetricCard
          title="Malicious Verdicts"
          value={maliciousCount.toString()}
          subtitle="Confirmed high-risk findings"
          icon={AlertTriangle}
          color="rose"
        />
        <MetricCard
          title="GNN Analysis Modules"
          value="4 Models"
          subtitle="URLInfraGNN + 3 Malware Nets"
          icon={BrainCircuit}
          color="violet"
        />
      </div>

      {/* Visual Analytics Row: Threat Risk Distribution & Category Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Risk Distribution Chart */}
        <div className="soc-card p-5 lg:col-span-1 flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-sm font-semibold text-slate-100 font-mono">Risk Level Distribution</h3>
              <p className="text-[11px] text-slate-400">Aggregated from session detections</p>
            </div>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>

          <div className="flex-1 flex flex-col items-center justify-center min-h-[220px]">
            {totalScans === 0 ? (
              <EmptyState
                title="No Scans Logged"
                description="Perform a URL or malware scan to view real risk distributions."
                actionLabel="Start Scan"
                onAction={() => navigate('/threat-analyzer')}
              />
            ) : (
              <div className="w-full h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={riskDistributionData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      innerRadius={45}
                      outerRadius={75}
                      paddingAngle={4}
                    >
                      {riskDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#090e1c',
                        borderColor: '#1e293b',
                        borderRadius: '8px',
                        fontSize: '12px'
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-3 text-xs font-mono">
                  {riskDistributionData.map((d, i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                      <span className="text-slate-300">{d.name}: {d.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Category Breakdown Chart */}
        <div className="soc-card p-5 lg:col-span-2 flex flex-col">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-sm font-semibold text-slate-100 font-mono">Classification Breakdown</h3>
              <p className="text-[11px] text-slate-400">Benign vs Phishing vs Defacement vs Malware</p>
            </div>
            <Shield className="w-4 h-4 text-violet-400" />
          </div>

          <div className="flex-1 flex flex-col justify-center min-h-[220px]">
            {categoryChartData.length === 0 ? (
              <EmptyState
                title="No Threat Classifications Yet"
                description="Category distributions will populate automatically as URL and binary predictions are evaluated."
              />
            ) : (
              <div className="w-full h-52">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={categoryChartData} margin={{ top: 20, right: 20, left: -20, bottom: 5 }}>
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#090e1c',
                        borderColor: '#1e293b',
                        borderRadius: '8px',
                        fontSize: '12px'
                      }}
                    />
                    <Bar dataKey="count" fill="#00f2fe" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Threats Table & AI Models Health */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Scans Table */}
        <div className="soc-card p-5 lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-semibold text-slate-100 font-mono">Recent Threat Activity</h3>
              <p className="text-[11px] text-slate-400">Chronological telemetry of analyzed artifacts</p>
            </div>
            <button
              onClick={() => navigate('/scan-history')}
              className="text-xs text-cyan-400 hover:underline flex items-center gap-1 font-mono"
            >
              <span>View All History</span>
              <ArrowRight className="w-3 h-3" />
            </button>
          </div>

          {history.length === 0 ? (
            <EmptyState
              title="No Recent Scans"
              description="Analyze a URL or upload a binary in the Threat Analyzer to see live telemetry."
              actionLabel="Launch Threat Analyzer"
              onAction={() => navigate('/threat-analyzer')}
            />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 font-mono text-[11px]">
                    <th className="pb-2.5">Target</th>
                    <th className="pb-2.5">Classification</th>
                    <th className="pb-2.5">Risk Level</th>
                    <th className="pb-2.5">Confidence</th>
                    <th className="pb-2.5">Timestamp</th>
                    <th className="pb-2.5 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {history.slice(0, 5).map((scan) => (
                    <tr key={scan.id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-2.5 pr-2 max-w-[200px] truncate text-slate-200">
                        {scan.url || scan.fileName}
                      </td>
                      <td className="py-2.5 pr-2">
                        <span className="text-cyan-300 font-medium">{scan.classification}</span>
                      </td>
                      <td className="py-2.5 pr-2">
                        <StatusBadge status={scan.riskLevel} size="sm" />
                      </td>
                      <td className="py-2.5 pr-2 text-slate-300">
                        {scan.confidence !== null && scan.confidence !== undefined
                          ? `${(scan.confidence * (scan.confidence <= 1 ? 100 : 1)).toFixed(1)}%`
                          : 'N/A'}
                      </td>
                      <td className="py-2.5 pr-2 text-slate-400 text-[11px]">
                        {formatTimestamp(scan.timestamp)}
                      </td>
                      <td className="py-2.5 text-right">
                        <button
                          onClick={() => navigate('/explainable-ai')}
                          className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 hover:text-cyan-300 text-[10px]"
                        >
                          XAI
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Security Analysis Module Status */}
        <div className="soc-card p-5 flex flex-col space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div>
              <h3 className="text-sm font-semibold text-slate-100 font-mono">Module Status</h3>
              <p className="text-[11px] text-slate-400">Architectural pipeline engines</p>
            </div>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>

          <div className="space-y-3 flex-1 font-mono text-xs">
            {Object.entries(moduleStatus).map(([key, mod]) => (
              <div
                key={key}
                className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 flex items-center justify-between"
              >
                <div className="space-y-0.5">
                  <span className="font-medium text-slate-200 block text-xs">{mod.name}</span>
                  <span className="text-[10px] text-slate-400 block">
                    {key === 'urlGNN'
                      ? 'Graph Convolution + Attention'
                      : key === 'malmem'
                      ? 'Volatile Memory (CIC-MalMem-2022)'
                      : key === 'bodmas'
                      ? 'BODMAS & EMBER Static Binary'
                      : key === 'malapi'
                      ? 'MalAPI2019 Syscall Sequences'
                      : 'Integrated Gradients / TreeSHAP'}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-[10px] text-emerald-400 uppercase font-semibold">
                    {mod.status}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="p-3 rounded-lg bg-cyan-950/20 border border-cyan-500/20 text-[11px] text-cyan-300 flex items-center gap-2">
            <Sparkles className="w-4 h-4 flex-shrink-0" />
            <span>FastAPI Uvicorn connector ready for model inference calls.</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
