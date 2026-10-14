import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Shield,
  Activity,
  Globe,
  Binary,
  GitFork,
  Cpu,
  BrainCircuit,
  History,
  Bot,
  Settings,
  ChevronLeft,
  ChevronRight,
  Radio
} from 'lucide-react';
import { useBackend } from '../../context/BackendContext';

export const Sidebar = ({ isCollapsed, setIsCollapsed, mobileOpen, setMobileOpen }) => {
  const { isOnline, status } = useBackend();

  const navigationItems = [
    { name: 'Overview', path: '/', icon: Activity },
    { name: 'Threat Analyzer', path: '/threat-analyzer', icon: Shield, badge: 'Core' },
    { name: 'URL Intelligence', path: '/url-intelligence', icon: Globe },
    { name: 'Malware Analysis', path: '/malware-analysis', icon: Binary, badge: '3-Engines' },
    { name: 'Infrastructure Graph', path: '/infrastructure-graph', icon: GitFork, badge: 'GNN' },
    { name: 'Dynamic Analysis', path: '/dynamic-analysis', icon: Cpu },
    { name: 'Explainable AI', path: '/explainable-ai', icon: BrainCircuit, badge: 'XAI' },
    { name: 'Scan History', path: '/scan-history', icon: History },
    { name: 'AI Security Assistant', path: '/security-assistant', icon: Bot },
    { name: 'Settings & Status', path: '/settings', icon: Settings }
  ];

  const handleLinkClick = () => {
    if (setMobileOpen) setMobileOpen(false);
  };

  return (
    <>
      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/80 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex flex-col bg-[#090d1a] border-r border-slate-800/80 transition-all duration-300 ease-in-out
          ${isCollapsed ? 'w-20' : 'w-72'}
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Brand header */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-slate-800/80">
          <NavLink to="/" className="flex items-center gap-3 overflow-hidden" onClick={handleLinkClick}>
            <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500/20 to-violet-500/20 border border-cyan-500/40 text-cyan-400 shadow-glow-cyan flex-shrink-0">
              <Shield className="w-5 h-5" />
              <div className="absolute -bottom-1 -right-1 w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            </div>

            {!isCollapsed && (
              <div className="flex flex-col truncate">
                <span className="text-sm font-bold tracking-wider text-slate-100 font-mono flex items-center gap-1.5">
                  CYBER<span className="text-cyan-400">X</span>
                  <span className="text-[10px] px-1 py-0.2 bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 rounded">GNN-XAI</span>
                </span>
                <span className="text-[10px] text-slate-400 truncate">Threat Intel Platform</span>
              </div>
            )}
          </NavLink>

          {/* Desktop collapse toggle */}
          <button
            onClick={() => setIsCollapsed(!isCollapsed)}
            className="hidden lg:flex items-center justify-center w-7 h-7 rounded-lg bg-slate-800/60 hover:bg-slate-700/80 text-slate-400 hover:text-slate-100 border border-slate-700/50 transition-colors"
            title={isCollapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Live Backend Connection Chip in Sidebar */}
        <div className="px-3 py-2 border-b border-slate-800/50">
          <div
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-mono transition-colors ${
              isOnline
                ? 'bg-emerald-500/10 border border-emerald-500/20 text-emerald-400'
                : status === 'checking'
                ? 'bg-amber-500/10 border border-amber-500/20 text-amber-400'
                : 'bg-rose-500/10 border border-rose-500/20 text-rose-400'
            }`}
          >
            <Radio className={`w-3.5 h-3.5 flex-shrink-0 ${isOnline ? 'animate-pulse' : ''}`} />
            {!isCollapsed && (
              <div className="truncate flex-1">
                <span className="font-semibold block uppercase text-[10px]">
                  {status === 'checking' ? 'PROBING API...' : isOnline ? 'FASTAPI CONNECTED' : 'FASTAPI OFFLINE'}
                </span>
                <span className="text-[10px] text-slate-400 block truncate">
                  {isOnline ? 'Uvicorn Server Online' : 'Check port 8000'}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Navigation links */}
        <nav className="flex-1 px-3 py-4 space-y-1.5 overflow-y-auto">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                onClick={handleLinkClick}
                className={({ isActive }) => `
                  group relative flex items-center gap-3 px-3 py-2.5 rounded-xl text-xs font-medium transition-all duration-200
                  ${isActive
                    ? 'bg-gradient-to-r from-cyan-500/15 to-transparent text-cyan-300 border-l-2 border-cyan-400 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                  }
                `}
                title={isCollapsed ? item.name : undefined}
              >
                <Icon className="w-4 h-4 flex-shrink-0 transition-transform group-hover:scale-110" />

                {!isCollapsed && (
                  <div className="flex items-center justify-between flex-1 truncate">
                    <span className="truncate">{item.name}</span>
                    {item.badge && (
                      <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700/60">
                        {item.badge}
                      </span>
                    )}
                  </div>
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Project footer info */}
        {!isCollapsed && (
          <div className="p-3.5 m-3 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
            <p className="font-semibold text-slate-300 font-mono">EDAI College Project</p>
            <p className="text-[10px] text-slate-400 leading-tight">
              Explainable Zero-Day Malware & Malicious-URL Behavior Analysis
            </p>
          </div>
        )}
      </aside>
    </>
  );
};

export default Sidebar;
