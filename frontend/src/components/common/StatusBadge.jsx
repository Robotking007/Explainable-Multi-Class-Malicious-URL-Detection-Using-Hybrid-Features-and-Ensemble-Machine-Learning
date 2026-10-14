import React from 'react';
import { ShieldCheck, AlertTriangle, ShieldAlert, HelpCircle, Activity } from 'lucide-react';

export const StatusBadge = ({ status, size = 'md', showIcon = true, pulse = false }) => {
  const norm = (status || 'unknown').toLowerCase();

  let config = {
    label: status || 'Unknown',
    bg: 'bg-slate-800/80',
    text: 'text-slate-300',
    border: 'border-slate-700',
    dot: 'bg-slate-400',
    icon: HelpCircle,
    glow: ''
  };

  if (norm.includes('safe') || norm.includes('benign') || norm === 'low') {
    config = {
      label: 'Benign / Safe',
      bg: 'bg-emerald-500/10',
      text: 'text-emerald-400',
      border: 'border-emerald-500/30',
      dot: 'bg-emerald-400',
      icon: ShieldCheck,
      glow: 'shadow-[0_0_12px_rgba(16,185,129,0.2)]'
    };
  } else if (norm.includes('suspicious') || norm === 'medium') {
    config = {
      label: 'Suspicious',
      bg: 'bg-amber-500/10',
      text: 'text-amber-400',
      border: 'border-amber-500/30',
      dot: 'bg-amber-400',
      icon: AlertTriangle,
      glow: 'shadow-[0_0_12px_rgba(245,158,11,0.2)]'
    };
  } else if (norm.includes('phish')) {
    config = {
      label: 'Phishing',
      bg: 'bg-orange-500/10',
      text: 'text-orange-400',
      border: 'border-orange-500/30',
      dot: 'bg-orange-400',
      icon: ShieldAlert,
      glow: 'shadow-[0_0_12px_rgba(249,115,22,0.2)]'
    };
  } else if (norm.includes('deface')) {
    config = {
      label: 'Defacement',
      bg: 'bg-purple-500/10',
      text: 'text-purple-400',
      border: 'border-purple-500/30',
      dot: 'bg-purple-400',
      icon: Activity,
      glow: 'shadow-[0_0_12px_rgba(168,85,247,0.2)]'
    };
  } else if (norm.includes('malware') || norm.includes('high') || norm.includes('critical') || norm.includes('malicious')) {
    config = {
      label: status?.toUpperCase() || 'MALICIOUS',
      bg: 'bg-rose-500/10',
      text: 'text-rose-400',
      border: 'border-rose-500/30',
      dot: 'bg-rose-500',
      icon: ShieldAlert,
      glow: 'shadow-[0_0_16px_rgba(239,68,68,0.25)]'
    };
  }

  const IconComponent = config.icon;
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : size === 'lg' ? 'px-3.5 py-1.5 text-sm font-semibold' : 'px-2.5 py-1 text-xs font-medium';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border ${config.bg} ${config.text} ${config.border} ${config.glow} ${sizeClasses}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot} ${pulse ? 'animate-ping' : ''}`} />
      {showIcon && <IconComponent className={size === 'sm' ? 'w-3 h-3' : 'w-3.5 h-3.5'} />}
      <span className="tracking-wide uppercase font-mono">{config.label}</span>
    </span>
  );
};

export default StatusBadge;
