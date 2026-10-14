import React from 'react';
import { AlertCircle, RefreshCw, ServerOff, Terminal } from 'lucide-react';

export const ErrorBanner = ({
  error,
  title = 'Analysis Pipeline Error',
  onRetry = null,
  compact = false
}) => {
  if (!error) return null;

  const errorMessage = typeof error === 'string' ? error : error.message || 'An error occurred during communication.';
  const isOffline = error?.isOffline || errorMessage.includes('Cannot connect to FastAPI backend');

  if (compact) {
    return (
      <div className="flex items-center justify-between p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">
        <div className="flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0 text-rose-400" />
          <span>{errorMessage}</span>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="flex items-center gap-1 text-rose-300 hover:text-white font-medium underline ml-2"
          >
            <RefreshCw className="w-3 h-3" />
            Retry
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="p-5 rounded-xl bg-gradient-to-r from-rose-950/30 to-slate-900/60 border border-rose-500/30 text-slate-200 my-4 shadow-glow-danger">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3.5">
          <div className="p-2 rounded-lg bg-rose-500/20 text-rose-400 mt-0.5">
            {isOffline ? <ServerOff className="w-5 h-5" /> : <AlertCircle className="w-5 h-5" />}
          </div>
          <div>
            <h4 className="text-sm font-semibold text-rose-300 font-mono flex items-center gap-2">
              <span>{title}</span>
              {error?.status && (
                <span className="px-1.5 py-0.5 text-[10px] bg-rose-900/50 rounded border border-rose-700/50">
                  HTTP {error.status}
                </span>
              )}
            </h4>
            <p className="text-xs text-slate-300 mt-1.5 leading-relaxed">{errorMessage}</p>

            {isOffline && (
              <div className="mt-3 p-2.5 rounded-lg bg-slate-950/70 border border-slate-800 text-[11px] font-mono text-slate-400 flex items-center gap-2">
                <Terminal className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
                <span>Tip: Verify FastAPI server is running with: <code className="text-cyan-300">uvicorn main:app --reload --port 8000</code></span>
              </div>
            )}
          </div>
        </div>

        {onRetry && (
          <button
            onClick={onRetry}
            className="soc-btn-secondary text-xs px-3 py-1.5 flex items-center gap-1.5 flex-shrink-0"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Retry</span>
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorBanner;
