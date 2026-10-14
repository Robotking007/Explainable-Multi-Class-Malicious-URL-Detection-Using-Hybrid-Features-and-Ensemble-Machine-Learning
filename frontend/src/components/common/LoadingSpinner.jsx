import React from 'react';
import { Loader2, ShieldCheck, Cpu } from 'lucide-react';

export const LoadingSpinner = ({
  message = 'Executing AI Pipeline Inference...',
  subtext = 'Analyzing topology graphs & neural representations',
  step = null
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center my-6">
      <div className="relative flex items-center justify-center w-20 h-20 mb-4">
        {/* Outer rotating cyber ring */}
        <div className="absolute inset-0 rounded-full border-2 border-dashed border-cyan-500/40 animate-spin" style={{ animationDuration: '6s' }} />
        {/* Inner pulsing ring */}
        <div className="absolute inset-2 rounded-full border border-violet-500/30 animate-ping opacity-30" />
        {/* Center glowing radar dot */}
        <div className="w-12 h-12 rounded-full bg-gradient-to-tr from-cyan-500/20 to-blue-600/30 border border-cyan-400/50 flex items-center justify-center shadow-glow-cyan">
          <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
        </div>
      </div>

      <h4 className="text-base font-semibold text-slate-100 font-mono tracking-wide">{message}</h4>
      {subtext && <p className="text-xs text-slate-400 mt-1 max-w-sm">{subtext}</p>}

      {step && (
        <div className="mt-4 px-3 py-1 rounded bg-slate-900/80 border border-slate-800 text-[11px] font-mono text-cyan-300">
          STAGE: {step}
        </div>
      )}
    </div>
  );
};

export default LoadingSpinner;
