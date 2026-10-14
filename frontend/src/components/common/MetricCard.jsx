import React from 'react';

export const MetricCard = ({
  title,
  value,
  subtitle,
  icon: Icon,
  badge = null,
  color = 'cyan', // 'cyan' | 'violet' | 'emerald' | 'amber' | 'rose'
  trend = null
}) => {
  const colorMap = {
    cyan: {
      border: 'hover:border-cyan-500/30',
      iconBg: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
      glow: 'group-hover:shadow-glow-cyan'
    },
    violet: {
      border: 'hover:border-violet-500/30',
      iconBg: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
      glow: 'group-hover:shadow-glow-violet'
    },
    emerald: {
      border: 'hover:border-emerald-500/30',
      iconBg: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
      glow: 'group-hover:shadow-[0_0_20px_-3px_rgba(16,185,129,0.3)]'
    },
    amber: {
      border: 'hover:border-amber-500/30',
      iconBg: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
      glow: 'group-hover:shadow-[0_0_20px_-3px_rgba(245,158,11,0.3)]'
    },
    rose: {
      border: 'hover:border-rose-500/30',
      iconBg: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
      glow: 'group-hover:shadow-glow-danger'
    }
  };

  const style = colorMap[color] || colorMap.cyan;

  return (
    <div
      className={`group relative p-5 rounded-xl bg-[#0d1322]/90 border border-slate-800/80 backdrop-blur-md transition-all duration-300 ${style.border} ${style.glow}`}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-slate-400 tracking-wider uppercase">{title}</p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl font-bold font-mono text-slate-100 tracking-tight">{value}</span>
            {badge && <span>{badge}</span>}
          </div>
          {subtitle && <p className="text-xs text-slate-400 mt-1">{subtitle}</p>}
          {trend && (
            <div className="mt-2 flex items-center gap-1 text-xs">
              <span className={trend.isPositive ? 'text-emerald-400' : 'text-rose-400'}>{trend.value}</span>
              <span className="text-slate-400 text-[11px]">{trend.label}</span>
            </div>
          )}
        </div>

        {Icon && (
          <div className={`p-3 rounded-xl border ${style.iconBg} flex items-center justify-center transition-transform group-hover:scale-105`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>
    </div>
  );
};

export default MetricCard;
