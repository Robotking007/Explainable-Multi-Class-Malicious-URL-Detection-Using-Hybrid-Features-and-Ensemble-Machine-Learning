import React from 'react';
import { X, ShieldAlert, AlertTriangle, CheckCircle2, Server, Trash2 } from 'lucide-react';
import { useBackend } from '../../context/BackendContext';

export const NotificationsDrawer = ({ isOpen, onClose }) => {
  const { isOnline, latency } = useBackend();

  if (!isOpen) return null;

  const alerts = [
    {
      id: 1,
      type: 'critical',
      title: 'Zero-Day Behavioral Anomaly',
      desc: 'High-risk topological clustering detected on bulletproof ASN 44050.',
      time: '12m ago'
    },
    {
      id: 2,
      type: 'warning',
      title: 'Obfuscated Memory Injection',
      desc: 'CIC-MalMem-2022 heuristic flagged anomalous VAD tree allocations.',
      time: '34m ago'
    },
    {
      id: 3,
      type: 'info',
      title: 'FastAPI Backend Status',
      desc: isOnline
        ? `Uvicorn API server online with response latency ${latency || 42}ms.`
        : 'Backend server currently unreachable on default port 8000.',
      time: 'Just now'
    }
  ];

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Drawer */}
      <div className="relative w-full max-w-sm h-full bg-[#0d1322] border-l border-slate-800 shadow-2xl flex flex-col z-10 font-sans">
        {/* Drawer header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-slate-100 font-mono">SOC Alert Feed</h3>
          </div>
          <button onClick={onClose} className="p-1 rounded text-slate-400 hover:text-slate-100">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Alerts list */}
        <div className="flex-1 p-4 overflow-y-auto space-y-3">
          {alerts.map((alert) => (
            <div
              key={alert.id}
              className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-colors"
            >
              <div className="flex items-start gap-2.5">
                {alert.type === 'critical' ? (
                  <ShieldAlert className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                ) : alert.type === 'warning' ? (
                  <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                ) : (
                  <Server className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-200">{alert.title}</span>
                    <span className="text-[10px] text-slate-400 font-mono">{alert.time}</span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1 leading-normal">{alert.desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-800 bg-slate-950/50 flex justify-between items-center text-xs">
          <span className="text-slate-400 text-[11px]">System: URLInfraGNN Node</span>
          <button onClick={onClose} className="text-cyan-400 hover:underline text-[11px]">
            Dismiss All
          </button>
        </div>
      </div>
    </div>
  );
};

export default NotificationsDrawer;
