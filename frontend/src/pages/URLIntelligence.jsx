import React, { useState } from 'react';
import {
  Globe,
  Search,
  Server,
  Layers,
  ShieldAlert,
  Copy,
  Check,
  ExternalLink,
  Calendar,
  MapPin,
  Cpu,
  ChevronDown,
  ChevronUp,
  Info
} from 'lucide-react';
import intelligenceService from '../services/intelligenceService';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorBanner from '../components/common/ErrorBanner';
import StatusBadge from '../components/common/StatusBadge';
import { copyToClipboard } from '../utils/helpers';

export const URLIntelligence = () => {
  const [targetUrl, setTargetUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [intelData, setIntelData] = useState(null);
  const [copiedKey, setCopiedKey] = useState(null);
  const [expandedSections, setExpandedSections] = useState({
    whois: true,
    network: true,
    dns: true,
    redirects: true
  });

  const handleCopy = async (key, text) => {
    const success = await copyToClipboard(text);
    if (success) {
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 2000);
    }
  };

  const toggleSection = (section) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const handleQuery = async (queryUrl = targetUrl) => {
    if (!queryUrl || !queryUrl.trim()) {
      setError('Please provide a URL or domain to query.');
      return;
    }

    setLoading(true);
    setError(null);
    setIntelData(null);

    try {
      const data = await intelligenceService.getUrlIntelligence(queryUrl.trim());
      setIntelData(data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const sampleTargets = [
    'https://login.microsoftonline.security-token-renewal.com',
    'http://185.220.101.5/admin/panel',
    'https://github.com/torvalds/linux'
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">RECONNAISSANCE</span>
            <span className="text-xs text-slate-400 font-mono">• Host & Network Telemetry</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            URL & Host Intelligence
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Deep-dive inspection into domain registration, WHOIS records, Autonomous System (ASN) prefix allocations, and redirect infrastructure.
          </p>
        </div>
      </div>

      {/* Query Bar */}
      <div className="soc-card p-5 space-y-3">
        <label htmlFor="intel-target-input" className="text-xs font-semibold text-slate-200 font-mono flex items-center gap-2">
          <Globe className="w-4 h-4 text-cyan-400" />
          <span>Enter Domain or Full URL</span>
        </label>
        <div className="flex flex-col sm:flex-row gap-3">
          <input
            id="intel-target-input"
            type="text"
            placeholder="e.g. account-verification.xyz or https://evil.com/login"
            value={targetUrl}
            onChange={(e) => setTargetUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
            disabled={loading}
            className="flex-1 bg-[#080d19] border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 font-mono focus:outline-none focus:border-cyan-400"
          />
          <button
            onClick={() => handleQuery()}
            disabled={loading || !targetUrl.trim()}
            className="soc-btn-primary px-6 py-2.5 text-xs font-mono"
          >
            <Search className="w-4 h-4" />
            <span>Lookup Intelligence</span>
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2 pt-2 text-[11px] font-mono text-slate-400">
          <span>Presets:</span>
          {sampleTargets.map((item, idx) => (
            <button
              key={idx}
              onClick={() => {
                setTargetUrl(item);
                handleQuery(item);
              }}
              className="px-2 py-0.5 rounded bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-slate-800"
            >
              {item}
            </button>
          ))}
        </div>
      </div>

      {loading && (
        <LoadingSpinner
          message="Performing WHOIS, ASN & Reverse DNS Lookups"
          subtext="Querying autonomous system registries and parsing certificate authorities..."
        />
      )}

      {error && (
        <ErrorBanner
          error={error}
          title="Intelligence Lookup Error"
          onRetry={() => handleQuery()}
        />
      )}

      {/* Result Display */}
      {intelData && !loading && (
        <div className="space-y-6">
          {/* Top Overview Card */}
          <div className="soc-card p-6 border-l-4 border-l-cyan-400">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <span className="text-[10px] font-mono text-slate-400 uppercase">Target Domain</span>
                <h3 className="text-xl font-bold font-mono text-slate-100 flex items-center gap-2">
                  <span>{intelData.domain}</span>
                  <button
                    onClick={() => handleCopy('domain', intelData.domain)}
                    className="p-1 rounded text-slate-400 hover:text-slate-200"
                    title="Copy Domain"
                  >
                    {copiedKey === 'domain' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  </button>
                </h3>
                <span className="text-xs text-slate-400 font-mono">Domain Age: {intelData.domainAge}</span>
              </div>

              <div className="flex flex-wrap gap-2">
                <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono">
                  <span className="text-slate-400 block text-[10px]">Resolved IP</span>
                  <span className="text-cyan-300 font-bold">{intelData.network?.ip}</span>
                </div>
                <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono">
                  <span className="text-slate-400 block text-[10px]">ASN Registry</span>
                  <span className="text-violet-300 font-bold">{intelData.network?.asn}</span>
                </div>
              </div>
            </div>
          </div>

          {/* 2-Column Grid: WHOIS vs Network Infrastructure */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* WHOIS Record Box */}
            <div className="soc-card p-5 space-y-4">
              <div
                className="flex items-center justify-between cursor-pointer pb-2 border-b border-slate-800"
                onClick={() => toggleSection('whois')}
              >
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-cyan-400" />
                  <h4 className="text-xs font-bold text-slate-200 font-mono uppercase">WHOIS Registration Info</h4>
                </div>
                {expandedSections.whois ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </div>

              {expandedSections.whois && (
                <div className="space-y-2.5 font-mono text-xs">
                  <div className="flex justify-between p-2 rounded bg-slate-900/80 border border-slate-800">
                    <span className="text-slate-400">Registrar:</span>
                    <span className="text-slate-200 font-medium">{intelData.whois?.registrar}</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-slate-900/80 border border-slate-800">
                    <span className="text-slate-400">Created:</span>
                    <span className="text-slate-200 font-medium">{intelData.whois?.creationDate}</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-slate-900/80 border border-slate-800">
                    <span className="text-slate-400">Expiration:</span>
                    <span className="text-slate-200 font-medium">{intelData.whois?.expirationDate}</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-slate-900/80 border border-slate-800">
                    <span className="text-slate-400">Country:</span>
                    <span className="text-slate-200 font-medium">{intelData.whois?.registrantCountry}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Network / ASN Box */}
            <div className="soc-card p-5 space-y-4">
              <div
                className="flex items-center justify-between cursor-pointer pb-2 border-b border-slate-800"
                onClick={() => toggleSection('network')}
              >
                <div className="flex items-center gap-2">
                  <Server className="w-4 h-4 text-violet-400" />
                  <h4 className="text-xs font-bold text-slate-200 font-mono uppercase">ASN & Network Routing</h4>
                </div>
                {expandedSections.network ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
              </div>

              {expandedSections.network && (
                <div className="space-y-2.5 font-mono text-xs">
                  <div className="flex justify-between p-2 rounded bg-slate-900/80 border border-slate-800">
                    <span className="text-slate-400">AS Name / ISP:</span>
                    <span className="text-slate-200 font-medium">{intelData.network?.asName}</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-slate-900/80 border border-slate-800">
                    <span className="text-slate-400">CIDR Prefix:</span>
                    <span className="text-slate-200 font-medium">{intelData.network?.prefix}</span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-slate-900/80 border border-slate-800">
                    <span className="text-slate-400">Geographic Location:</span>
                    <span className="text-slate-200 font-medium flex items-center gap-1">
                      <MapPin className="w-3 h-3 text-cyan-400" />
                      {intelData.network?.geo?.city}, {intelData.network?.geo?.country}
                    </span>
                  </div>
                  <div className="flex justify-between p-2 rounded bg-slate-900/80 border border-slate-800">
                    <span className="text-slate-400">IP Host Classification:</span>
                    <span className="text-cyan-300 font-medium">BGP Advertised IPv4</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default URLIntelligence;
