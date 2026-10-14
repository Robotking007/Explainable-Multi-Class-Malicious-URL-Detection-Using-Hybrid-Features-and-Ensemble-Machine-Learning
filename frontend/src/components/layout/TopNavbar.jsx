import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Menu,
  Search,
  RefreshCw,
  Bell,
  Shield,
  Server,
  Zap,
  ExternalLink,
  ChevronDown,
  User,
  Sliders
} from 'lucide-react';
import { useBackend } from '../../context/BackendContext';

export const TopNavbar = ({ setMobileOpen, onOpenNotifications }) => {
  const { status, isOnline, latency, apiUrl, refreshStatus } = useBackend();
  const [quickQuery, setQuickQuery] = useState('');
  const [showProfileMenu, setShowProfileMenu] = useState(false);
  const navigate = useNavigate();

  const handleQuickSearch = (e) => {
    e.preventDefault();
    if (!quickQuery.trim()) return;
    // Navigate to Threat Analyzer with search query parameter
    navigate(`/threat-analyzer?query=${encodeURIComponent(quickQuery.trim())}`);
    setQuickQuery('');
  };

  return (
    <header className="sticky top-0 z-30 flex items-center justify-between h-16 px-4 md:px-6 bg-[#090d1a]/90 backdrop-blur-md border-b border-slate-800/80">
      {/* Left side: Mobile menu toggle + Global quick scan bar */}
      <div className="flex items-center gap-3 flex-1 max-w-xl">
        <button
          onClick={() => setMobileOpen(true)}
          className="p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 lg:hidden"
          aria-label="Open Navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        <form onSubmit={handleQuickSearch} className="relative flex-1">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Quick Scan: Paste suspicious URL, IP, or Hash..."
            value={quickQuery}
            onChange={(e) => setQuickQuery(e.target.value)}
            className="w-full pl-9 pr-20 py-2 rounded-lg bg-slate-900/90 border border-slate-700/60 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/20 font-mono transition-all"
          />
          <button
            type="submit"
            className="absolute right-1.5 top-1/2 -translate-y-1/2 px-2.5 py-1 rounded bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-[10px] font-mono font-medium transition-all"
          >
            SCAN
          </button>
        </form>
      </div>

      {/* Right side: Real backend health pill, notifications, profile */}
      <div className="flex items-center gap-3">
        {/* Backend status pill */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 font-mono text-xs">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                status === 'checking'
                  ? 'bg-amber-400 animate-ping'
                  : isOnline
                  ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]'
                  : 'bg-rose-500'
              }`}
            />
            <span className="text-slate-300 font-medium text-[11px]">
              {status === 'checking'
                ? 'CHECKING'
                : isOnline
                ? 'FASTAPI 200'
                : 'OFFLINE'}
            </span>
          </div>

          {isOnline && latency !== null && (
            <span className="text-[10px] text-cyan-400 border-l border-slate-700 pl-2">
              {latency}ms
            </span>
          )}

          <button
            onClick={refreshStatus}
            title={`Refresh API Health (${apiUrl})`}
            className="text-slate-500 hover:text-cyan-400 transition-colors ml-1 p-0.5 rounded"
          >
            <RefreshCw className={`w-3 h-3 ${status === 'checking' ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Notifications button */}
        <button
          onClick={onOpenNotifications}
          className="relative p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 transition-colors"
          title="Security Alerts"
        >
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
        </button>

        {/* User profile / SOC Analyst badge */}
        <div className="relative">
          <button
            onClick={() => setShowProfileMenu(!showProfileMenu)}
            className="flex items-center gap-2.5 p-1.5 rounded-xl hover:bg-slate-800/50 transition-colors"
          >
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-violet-600 p-[1px]">
              <div className="w-full h-full bg-[#0d1322] rounded-[7px] flex items-center justify-center text-cyan-400">
                <User className="w-4 h-4" />
              </div>
            </div>
            <div className="hidden md:block text-left text-xs">
              <span className="block font-semibold text-slate-200 font-mono">SOC Analyst</span>
              <span className="block text-[10px] text-slate-400">College Demo</span>
            </div>
            <ChevronDown className="w-3.5 h-3.5 text-slate-500 hidden md:block" />
          </button>

          {showProfileMenu && (
            <div
              className="absolute right-0 mt-2 w-56 rounded-xl bg-[#0e1424] border border-slate-700/80 shadow-2xl py-2 z-50 text-xs font-mono"
              onMouseLeave={() => setShowProfileMenu(false)}
            >
              <div className="px-3 py-2 border-b border-slate-800">
                <p className="font-semibold text-slate-200">EDAI Research Workstation</p>
                <p className="text-[10px] text-cyan-400 break-all">{apiUrl}</p>
              </div>
              <button
                onClick={() => {
                  setShowProfileMenu(false);
                  navigate('/settings');
                }}
                className="w-full px-3 py-2 text-left text-slate-300 hover:bg-slate-800/60 flex items-center gap-2"
              >
                <Sliders className="w-3.5 h-3.5 text-slate-400" />
                <span>Backend Settings & Models</span>
              </button>
              <button
                onClick={() => {
                  setShowProfileMenu(false);
                  navigate('/threat-analyzer');
                }}
                className="w-full px-3 py-2 text-left text-slate-300 hover:bg-slate-800/60 flex items-center gap-2"
              >
                <Shield className="w-3.5 h-3.5 text-cyan-400" />
                <span>Launch Threat Analyzer</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default TopNavbar;
