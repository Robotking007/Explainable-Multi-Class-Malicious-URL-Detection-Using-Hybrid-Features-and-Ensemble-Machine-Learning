import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldAlert, ArrowLeft } from 'lucide-react';

export const NotFound = () => {
  const navigate = useNavigate();

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] text-center p-8">
      <div className="w-16 h-16 rounded-2xl bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 mb-4 shadow-glow-danger">
        <ShieldAlert className="w-8 h-8" />
      </div>
      <h1 className="text-3xl font-extrabold font-mono text-slate-100">404 - Access Denied / Route Missing</h1>
      <p className="text-sm text-slate-400 max-w-md mt-2 font-mono">
        The requested threat telemetry routing does not exist or has been sequestered by SOC policy.
      </p>
      <button
        onClick={() => navigate('/')}
        className="soc-btn-primary mt-6 text-xs px-5 py-2.5 font-mono flex items-center gap-2"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Return to SOC Dashboard</span>
      </button>
    </div>
  );
};

export default NotFound;
