import React, { useState } from 'react';
import {
  History,
  Search,
  Filter,
  Download,
  Trash2,
  ExternalLink,
  Calendar,
  Layers,
  Shield,
  X,
  Copy,
  Check,
  Info
} from 'lucide-react';
import { useScans } from '../context/ScanContext';
import StatusBadge from '../components/common/StatusBadge';
import EmptyState from '../components/common/EmptyState';
import { formatTimestamp, exportToCSV, copyToClipboard } from '../utils/helpers';

export const ScanHistory = () => {
  const { history, clearAllHistory, historySource } = useScans();

  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState('ALL'); // 'ALL' | 'URL' | 'MALWARE'
  const [filterRisk, setFilterRisk] = useState('ALL'); // 'ALL' | 'MALICIOUS' | 'SUSPICIOUS' | 'BENIGN'
  const [selectedScan, setSelectedScan] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 8;

  // Filter scans
  const filteredScans = history.filter((scan) => {
    const target = (scan.url || scan.fileName || scan.id || '').toLowerCase();
    const matchesSearch = target.includes(searchTerm.toLowerCase());

    const matchesType =
      filterType === 'ALL' || (scan.type || 'URL').toUpperCase() === filterType;

    const risk = (scan.riskLevel || '').toUpperCase();
    const matchesRisk =
      filterRisk === 'ALL' ||
      (filterRisk === 'MALICIOUS' && (risk.includes('MALICIOUS') || risk.includes('HIGH') || risk.includes('CRITICAL'))) ||
      (filterRisk === 'SUSPICIOUS' && risk.includes('SUSPICIOUS')) ||
      (filterRisk === 'BENIGN' && (risk.includes('SAFE') || risk.includes('BENIGN')));

    return matchesSearch && matchesType && matchesRisk;
  });

  const totalPages = Math.ceil(filteredScans.length / itemsPerPage) || 1;
  const paginatedScans = filteredScans.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const handleExport = () => {
    if (!history.length) return;
    const exportRows = history.map((s) => ({
      ID: s.id,
      Type: s.type || 'URL',
      Target: s.url || s.fileName,
      Classification: s.classification,
      RiskLevel: s.riskLevel,
      Confidence: s.confidence,
      Timestamp: s.timestamp
    }));
    exportToCSV(`cyberx_threat_telemetry_${Date.now()}.csv`, exportRows);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">AUDIT TELEMETRY</span>
            <span className="text-xs text-slate-400 font-mono">• Chronological Logs</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            Scan History & Audit Trail
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Review past URL detections, binary scans, and model verdicts with CSV export capability.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleExport}
            disabled={history.length === 0}
            className="soc-btn-secondary text-xs px-3.5 py-2 flex items-center gap-2 font-mono"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Export CSV</span>
          </button>
          <button
            onClick={clearAllHistory}
            disabled={history.length === 0}
            className="p-2 rounded-lg text-slate-400 hover:text-rose-400 border border-slate-800 hover:border-rose-500/30 transition-colors"
            title="Clear All Scan Logs"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Storage Source Notice */}
      <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400 flex items-center justify-between font-mono">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 text-cyan-400" />
          <span>
            Storage Source: {historySource === 'backend' ? 'PostgreSQL/FastAPI Backend DB' : 'Session Browser Storage'}
          </span>
        </div>
        <span className="text-[11px] text-slate-500">{history.length} Total Telemetry Records</span>
      </div>

      {/* Filters & Search Toolbar */}
      <div className="soc-card p-4 flex flex-col md:flex-row items-center justify-between gap-3 font-mono text-xs">
        <div className="relative flex-1 w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search by URL, filename, or scan identifier..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full pl-9 pr-4 py-2 rounded-xl bg-[#080d19] border border-slate-700/80 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400"
          />
        </div>

        <div className="flex flex-wrap items-center gap-2 w-full md:w-auto">
          {/* Filter Type */}
          <select
            value={filterType}
            onChange={(e) => {
              setFilterType(e.target.value);
              setCurrentPage(1);
            }}
            className="bg-[#080d19] border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-400 font-mono"
          >
            <option value="ALL">All Types</option>
            <option value="URL">URL Scans</option>
            <option value="MALWARE">Malware Scans</option>
          </select>

          {/* Filter Risk */}
          <select
            value={filterRisk}
            onChange={(e) => {
              setFilterRisk(e.target.value);
              setCurrentPage(1);
            }}
            className="bg-[#080d19] border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-400 font-mono"
          >
            <option value="ALL">All Severities</option>
            <option value="MALICIOUS">Malicious</option>
            <option value="SUSPICIOUS">Suspicious</option>
            <option value="BENIGN">Benign</option>
          </select>
        </div>
      </div>

      {/* History Table */}
      <div className="soc-card overflow-hidden">
        {filteredScans.length === 0 ? (
          <EmptyState
            icon={History}
            title="No Matching Records"
            description="No scan records matched your filter criteria or no scans have been performed yet."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-900/60 text-slate-400 text-[11px]">
                  <th className="p-3.5">Type</th>
                  <th className="p-3.5">Target Artifact</th>
                  <th className="p-3.5">Classification</th>
                  <th className="p-3.5">Risk Level</th>
                  <th className="p-3.5">Confidence</th>
                  <th className="p-3.5">Timestamp</th>
                  <th className="p-3.5 text-right">Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {paginatedScans.map((scan) => (
                  <tr key={scan.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="p-3.5">
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700 text-[10px]">
                        {scan.type || 'URL'}
                      </span>
                    </td>
                    <td className="p-3.5 max-w-xs truncate text-slate-200">
                      {scan.url || scan.fileName}
                    </td>
                    <td className="p-3.5 text-slate-300 font-semibold">{scan.classification}</td>
                    <td className="p-3.5">
                      <StatusBadge status={scan.riskLevel} size="sm" />
                    </td>
                    <td className="p-3.5 text-slate-300">
                      {scan.confidence !== null && scan.confidence !== undefined
                        ? `${(scan.confidence * (scan.confidence <= 1 ? 100 : 1)).toFixed(1)}%`
                        : 'N/A'}
                    </td>
                    <td className="p-3.5 text-slate-400 text-[11px]">
                      {formatTimestamp(scan.timestamp)}
                    </td>
                    <td className="p-3.5 text-right">
                      <button
                        onClick={() => setSelectedScan(scan)}
                        className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-300 hover:text-cyan-200 text-[11px]"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination controls */}
        {totalPages > 1 && (
          <div className="p-3.5 border-t border-slate-800 flex items-center justify-between text-xs font-mono text-slate-400">
            <span>
              Page {currentPage} of {totalPages}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 rounded bg-slate-800 text-slate-300 disabled:opacity-40"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 rounded bg-slate-800 text-slate-300 disabled:opacity-40"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Modal / Details Drawer for Selected Scan */}
      {selectedScan && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="fixed inset-0 bg-black/80 backdrop-blur-sm" onClick={() => setSelectedScan(null)} />

          <div className="relative w-full max-w-xl bg-[#0d1322] border border-cyan-500/30 rounded-2xl p-6 shadow-2xl space-y-4 font-mono text-xs z-10">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-slate-100">Scan Telemetry Breakdown</h3>
              </div>
              <button onClick={() => setSelectedScan(null)} className="p-1 text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3">
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase">Target Artifact</span>
                <span className="text-cyan-300 font-bold break-all text-xs">
                  {selectedScan.url || selectedScan.fileName}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 block text-[10px] uppercase">Classification</span>
                  <span className="text-slate-200 font-bold">{selectedScan.classification}</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 block text-[10px] uppercase">Severity</span>
                  <StatusBadge status={selectedScan.riskLevel} size="sm" />
                </div>
              </div>

              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-slate-400 block text-[10px] uppercase">Scan ID & Timestamp</span>
                <span className="text-slate-300 block">{selectedScan.id}</span>
                <span className="text-slate-500 text-[11px] block">{formatTimestamp(selectedScan.timestamp)}</span>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedScan(null)}
                className="soc-btn-secondary text-xs px-4 py-2"
              >
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ScanHistory;
