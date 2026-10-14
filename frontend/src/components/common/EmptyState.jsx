import React from 'react';
import { Shield, ArrowRight } from 'lucide-react';

export const EmptyState = ({
  icon: Icon = Shield,
  title = 'No Data Available',
  description = 'There are no active records or analysis findings to display at this moment.',
  actionLabel = null,
  onAction = null,
  secondaryAction = null
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center border border-dashed border-slate-800 rounded-2xl bg-slate-950/40 my-4">
      <div className="w-14 h-14 rounded-2xl bg-slate-900/80 border border-slate-800 flex items-center justify-center mb-4 text-cyan-400 shadow-inner">
        <Icon className="w-7 h-7" />
      </div>
      <h3 className="text-base font-semibold text-slate-200 tracking-tight">{title}</h3>
      <p className="text-sm text-slate-400 max-w-md mt-1.5 leading-relaxed">{description}</p>
      
      {(actionLabel || secondaryAction) && (
        <div className="flex items-center gap-3 mt-6">
          {actionLabel && (
            <button
              onClick={onAction}
              className="soc-btn-primary text-xs px-4 py-2"
            >
              <span>{actionLabel}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
          {secondaryAction}
        </div>
      )}
    </div>
  );
};

export default EmptyState;
