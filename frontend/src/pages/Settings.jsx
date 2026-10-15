import React, { useState, useEffect, useMemo } from 'react';
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
  Check,
  CloudDownload,
  HardDrive,
  Zap,
  Search,
  Filter,
  Sparkles,
  Box,
  Flame,
  Radio,
  FileCode,
  Network,
  Activity
} from 'lucide-react';
import { useBackend } from '../context/BackendContext';
import {
  getHfModelStatus,
  loadAllModelsFromHf,
  loadSpecificModelFromHf,
  reloadModelMemoryCache,
} from '../services/modelService';

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

  // Hugging Face Hub State
  const [hfData, setHfData] = useState(null);
  const [hfLoading, setHfLoading] = useState(true);
  const [hfError, setHfError] = useState(null);
  const [isSyncingAll, setIsSyncingAll] = useState(false);
  const [isWarmingCache, setIsWarmingCache] = useState(false);
  const [syncingSingle, setSyncingSingle] = useState({});
  const [syncFeedback, setSyncFeedback] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('ALL');

  // Fetch HF Model Status
  const fetchHfStatus = async () => {
    setHfLoading(true);
    setHfError(null);
    try {
      const data = await getHfModelStatus();
      setHfData(data);
    } catch (err) {
      console.error('Failed to query HF model status:', err);
      setHfError(err.message || 'Unable to retrieve Hugging Face model status.');
    } finally {
      setHfLoading(false);
    }
  };

  useEffect(() => {
    if (isOnline) {
      fetchHfStatus();
    }
  }, [isOnline]);

  const handleSaveUrl = (e) => {
    e.preventDefault();
    if (!inputUrl.trim()) return;
    updateApiUrl(inputUrl.trim());
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 2500);
  };

  // One-Click Pull & Load All Models
  const handleSyncAll = async (forceDownload = false) => {
    setIsSyncingAll(true);
    setSyncFeedback(null);
    try {
      const res = await loadAllModelsFromHf(forceDownload);
      setSyncFeedback({
        type: 'success',
        title: 'Hugging Face Synchronization Successful',
        message: res.message || `Successfully synced models from ${res.repo_id}`,
        details: res,
      });
      await fetchHfStatus();
    } catch (err) {
      setSyncFeedback({
        type: 'error',
        title: 'Hugging Face Pull Failed',
        message: err.message || 'Could not complete Hugging Face model pull.',
      });
    } finally {
      setIsSyncingAll(false);
    }
  };

  // Pull Single Model Checkpoint
  const handleSyncSingle = async (filename, forceDownload = false) => {
    setSyncingSingle((prev) => ({ ...prev, [filename]: true }));
    try {
      const res = await loadSpecificModelFromHf(filename, forceDownload);
      setSyncFeedback({
        type: 'success',
        title: `Model Checkpoint Loaded: ${filename}`,
        message: res.message || `Successfully downloaded ${filename} from Hugging Face Hub.`,
        details: res,
      });
      await fetchHfStatus();
    } catch (err) {
      setSyncFeedback({
        type: 'error',
        title: `Failed to Load ${filename}`,
        message: err.message || `Failed to download ${filename} from Hugging Face.`,
      });
    } finally {
      setSyncingSingle((prev) => ({ ...prev, [filename]: false }));
    }
  };

  // Warm In-Memory RAM Cache
  const handleWarmCache = async () => {
    setIsWarmingCache(true);
    try {
      const res = await reloadModelMemoryCache();
      setSyncFeedback({
        type: 'success',
        title: 'In-Memory Pipeline Cache Reloaded',
        message: res.message,
        details: res,
      });
      await fetchHfStatus();
    } catch (err) {
      setSyncFeedback({
        type: 'error',
        title: 'Cache Reload Failed',
        message: err.message || 'Failed to warm RAM caches.',
      });
    } finally {
      setIsWarmingCache(false);
    }
  };

  // Filtered Models
  const filteredModels = useMemo(() => {
    if (!hfData?.models) return [];
    return hfData.models.filter((m) => {
      const matchesCategory =
        selectedCategory === 'ALL' || m.category.toLowerCase().includes(selectedCategory.toLowerCase());
      const q = searchQuery.toLowerCase().trim();
      const matchesSearch =
        !q ||
        m.name.toLowerCase().includes(q) ||
        m.filename.toLowerCase().includes(q) ||
        m.category.toLowerCase().includes(q) ||
        m.engine.toLowerCase().includes(q) ||
        m.description.toLowerCase().includes(q);
      return matchesCategory && matchesSearch;
    });
  }, [hfData, selectedCategory, searchQuery]);

  const categories = useMemo(() => {
    if (!hfData?.categories) return ['ALL'];
    return ['ALL', ...hfData.categories];
  }, [hfData]);

  const getEngineBadgeColor = (engine) => {
    if (engine.includes('PyTorch') || engine.includes('PyG')) return 'border-orange-500/40 bg-orange-500/10 text-orange-400';
    if (engine.includes('Scikit-Learn') || engine.includes('Random Forest')) return 'border-cyan-500/40 bg-cyan-500/10 text-cyan-400';
    if (engine.includes('XGBoost')) return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400';
    if (engine.includes('SHAP')) return 'border-violet-500/40 bg-violet-500/10 text-violet-400';
    return 'border-slate-700 bg-slate-800 text-slate-300';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">SYSTEM ORCHESTRATION</span>
            <span className="text-xs text-slate-400 font-mono">• Model Hub & Telemetry</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            Settings & Model Management Hub
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Synchronize, load, and warm up all 31 deep learning, graph neural network, and classical ensemble model checkpoints from Hugging Face Hub directly in the frontend.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={fetchHfStatus}
            disabled={hfLoading}
            className="soc-btn-secondary text-xs px-3.5 py-2 font-mono flex items-center gap-2"
            title="Refresh Hugging Face Model Status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${hfLoading ? 'animate-spin' : ''}`} />
            <span>Check HF Status</span>
          </button>
          <button
            onClick={refreshStatus}
            className="soc-btn-secondary text-xs px-3.5 py-2 font-mono flex items-center gap-2"
          >
            <Radio className={`w-3.5 h-3.5 ${status === 'checking' ? 'animate-spin' : 'text-cyan-400'}`} />
            <span>Ping Backend</span>
          </button>
        </div>
      </div>

      {/* Sync Feedback Alert */}
      {syncFeedback && (
        <div
          className={`p-4 rounded-xl border font-mono text-xs flex items-start gap-3 transition-all ${
            syncFeedback.type === 'success'
              ? 'bg-emerald-950/40 border-emerald-500/40 text-emerald-300'
              : 'bg-rose-950/40 border-rose-500/40 text-rose-300'
          }`}
        >
          {syncFeedback.type === 'success' ? (
            <CheckCircle2 className="w-5 h-5 flex-shrink-0 text-emerald-400 mt-0.5" />
          ) : (
            <AlertCircle className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" />
          )}
          <div className="flex-1 space-y-1">
            <p className="font-bold">{syncFeedback.title}</p>
            <p className="text-[11px] text-slate-300 font-sans">{syncFeedback.message}</p>
            {syncFeedback.details?.warmed_in_memory && (
              <div className="pt-2 flex flex-wrap gap-1.5">
                {syncFeedback.details.warmed_in_memory.map((pipe, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[10px]"
                  >
                    Active: {pipe}
                  </span>
                ))}
              </div>
            )}
          </div>
          <button
            onClick={() => setSyncFeedback(null)}
            className="text-slate-400 hover:text-slate-100 text-xs px-2 py-1"
          >
            ✕
          </button>
        </div>
      )}

      {/* Hugging Face Model Hub Management Hero Card */}
      <div className="soc-card p-6 space-y-6 relative overflow-hidden border-cyan-500/30 bg-gradient-to-br from-[#0c1324] via-[#090e1a] to-[#070b14]">
        {/* Decorative corner glow */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-amber-500/20 to-orange-500/20 border border-amber-500/30 flex items-center justify-center text-amber-400 shadow-[0_0_15px_rgba(245,158,11,0.15)] flex-shrink-0">
              <span className="text-2xl" role="img" aria-label="Hugging Face">🤗</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold font-mono text-slate-100 flex items-center gap-2">
                  Hugging Face Model Registry & Sync
                </h2>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                  {hfData?.repo_id || 'siddudavant/cyberxai-models'}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Centralized weight storage on Hugging Face Hub with zero-cost local caching and in-memory hot-loading.
              </p>
            </div>
          </div>

          {/* Quick CTA Actions */}
          <div className="flex flex-wrap items-center gap-2">
            <a
              href={hfData?.repo_url || 'https://huggingface.co/siddudavant/cyberxai-models'}
              target="_blank"
              rel="noreferrer"
              className="soc-btn-secondary text-xs px-3 py-2 font-mono flex items-center gap-1.5"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              <span>View HF Repo</span>
            </a>

            <button
              onClick={handleWarmCache}
              disabled={isWarmingCache || isSyncingAll}
              className="soc-btn-secondary text-xs px-3 py-2 font-mono flex items-center gap-1.5 text-cyan-400 hover:text-cyan-300"
              title="Invalidate & warm RAM neural pipelines"
            >
              <Flame className={`w-3.5 h-3.5 ${isWarmingCache ? 'animate-bounce text-amber-400' : ''}`} />
              <span>{isWarmingCache ? 'Warming RAM...' : 'Warm RAM Cache'}</span>
            </button>

            <button
              onClick={() => handleSyncAll(false)}
              disabled={isSyncingAll || !isOnline}
              className="soc-btn-primary text-xs px-4 py-2 font-mono flex items-center gap-2 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold shadow-[0_0_20px_rgba(6,182,212,0.3)] transition-all"
            >
              <CloudDownload className={`w-4 h-4 ${isSyncingAll ? 'animate-spin' : ''}`} />
              <span>{isSyncingAll ? 'Loading Models from HF...' : 'Sync & Load All Models from Hugging Face'}</span>
            </button>
          </div>
        </div>

        {/* Real-time KPI Stats Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 font-mono text-xs">
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 block uppercase tracking-wider">Total Checkpoints</span>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold text-slate-100">
                {hfData?.total_remote_models ?? 31}
              </span>
              <span className="text-[10px] text-cyan-400">Hub Artifacts</span>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 block uppercase tracking-wider">Locally Cached</span>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold text-emerald-400">
                {hfData?.total_cached_locally ?? 31} / {hfData?.total_remote_models ?? 31}
              </span>
              <span className="text-[10px] text-emerald-500 font-bold">
                {hfData?.total_remote_models
                  ? `${Math.round((hfData.total_cached_locally / hfData.total_remote_models) * 100)}%`
                  : '100%'}
              </span>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 block uppercase tracking-wider">Active in Memory (RAM)</span>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold text-cyan-400">
                {hfData?.total_in_memory ?? 21}
              </span>
              <span className="flex items-center gap-1 text-[10px] text-cyan-300">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                Hot Loaded
              </span>
            </div>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <span className="text-[10px] text-slate-400 block uppercase tracking-wider">Storage Footprint</span>
            <div className="flex items-baseline gap-2">
              <span className="text-xl font-bold text-violet-400">
                {hfData?.total_storage_size ?? '1.09 GB'}
              </span>
              <span className="text-[10px] text-slate-400">Local Disk</span>
            </div>
          </div>
        </div>

        {/* Filter & Search Bar */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pt-2">
          {/* Category Tabs */}
          <div className="flex items-center gap-1 overflow-x-auto pb-1 sm:pb-0 font-mono text-[11px] scrollbar-thin">
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg whitespace-nowrap transition-all ${
                  selectedCategory === cat
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold'
                    : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 border border-slate-800/80'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Search Input */}
          <div className="relative min-w-[220px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search 31 checkpoints..."
              className="w-full bg-[#080d19] border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-400 font-mono"
            />
          </div>
        </div>

        {/* Models Table / List */}
        <div className="overflow-x-auto rounded-xl border border-slate-800/80 bg-[#080d19]/80 font-mono text-xs">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/60 text-slate-400 text-[11px]">
                <th className="py-3 px-4 font-semibold">Model Checkpoint & Artifact</th>
                <th className="py-3 px-4 font-semibold">Architecture / Engine</th>
                <th className="py-3 px-4 font-semibold">Category</th>
                <th className="py-3 px-4 font-semibold">Size</th>
                <th className="py-3 px-4 font-semibold">Cache Status</th>
                <th className="py-3 px-4 font-semibold">RAM Cache</th>
                <th className="py-3 px-4 font-semibold text-right">Hub Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {filteredModels.length > 0 ? (
                filteredModels.map((m) => {
                  const isSyncingThis = syncingSingle[m.filename];
                  return (
                    <tr key={m.filename} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-3 px-4">
                        <div className="flex flex-col">
                          <span className="font-bold text-slate-100 flex items-center gap-1.5">
                            {m.name}
                          </span>
                          <span className="text-[10px] text-cyan-400/80 break-all">{m.filename}</span>
                          <span className="text-[10px] text-slate-500 font-sans mt-0.5 line-clamp-1">
                            {m.description}
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`inline-block px-2 py-0.5 rounded text-[10px] border font-semibold ${getEngineBadgeColor(
                            m.engine
                          )}`}
                        >
                          {m.engine}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-[11px] text-slate-400">{m.category}</span>
                      </td>
                      <td className="py-3 px-4">
                        <span className="text-slate-300 font-semibold">{m.size_human}</span>
                      </td>
                      <td className="py-3 px-4">
                        {m.cached_locally ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-[10px] font-bold">
                            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                            CACHED
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30 text-[10px] font-bold">
                            <CloudDownload className="w-3 h-3 text-amber-400" />
                            REMOTE
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4">
                        {m.in_memory ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 text-[10px] font-bold">
                            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                            ACTIVE
                          </span>
                        ) : (
                          <span className="text-[10px] text-slate-500">STANDBY</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            onClick={() => handleSyncSingle(m.filename, true)}
                            disabled={isSyncingThis || isSyncingAll}
                            className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 hover:border-cyan-400/50 text-[10px] font-mono flex items-center gap-1 transition-all"
                            title={`Re-pull ${m.filename} from Hugging Face`}
                          >
                            <CloudDownload className={`w-3 h-3 ${isSyncingThis ? 'animate-spin text-cyan-400' : ''}`} />
                            <span>{isSyncingThis ? 'Pulling...' : m.cached_locally ? 'Re-pull' : 'Pull'}</span>
                          </button>
                          <a
                            href={m.hf_url}
                            target="_blank"
                            rel="noreferrer"
                            className="p-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-cyan-300 border border-slate-700 transition-colors"
                            title="Inspect artifact on Hugging Face"
                          >
                            <ExternalLink className="w-3 h-3" />
                          </a>
                        </div>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan="7" className="py-8 text-center text-slate-500">
                    {hfLoading ? (
                      <div className="flex items-center justify-center gap-2">
                        <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
                        <span>Querying Hugging Face Model Hub...</span>
                      </div>
                    ) : (
                      <span>No model checkpoints found matching your criteria.</span>
                    )}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
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
              <h3 className="text-sm font-bold font-mono text-slate-100">Live API Health Contract</h3>
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
                <p className="text-[10px] text-slate-600 mt-1">Start backend with: uvicorn api.main:app --reload</p>
              </div>
            )}
          </div>

          <div className="text-[11px] text-slate-400 font-sans">
            Health checks verify root <code className="text-cyan-300">/health</code> or <code className="text-cyan-300">/api/health</code> endpoint responsiveness.
          </div>
        </div>
      </div>

      {/* Application Meta */}
      <div className="soc-card p-6 space-y-2 font-mono text-xs">
        <h4 className="font-bold text-slate-200">Project Architecture & Metadata</h4>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2 text-slate-400">
          <div>
            <span className="block text-[10px] text-slate-500">PROJECT</span>
            <span className="text-slate-300">Explainable Multi-Class Malicious URL & Malware Platform</span>
          </div>
          <div>
            <span className="block text-[10px] text-slate-500">HUGGING FACE MODEL HUB</span>
            <span className="text-cyan-300">siddudavant/cyberxai-models (31 checkpoints)</span>
          </div>
          <div>
            <span className="block text-[10px] text-slate-500">TECH STACK</span>
            <span className="text-slate-300">React 19 + PyTorch DL + PyG GNN + TreeSHAP</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
