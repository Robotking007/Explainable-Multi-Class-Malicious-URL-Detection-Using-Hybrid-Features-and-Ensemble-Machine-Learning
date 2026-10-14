// Utility helper functions for CyberX Threat Intelligence

export const formatTimestamp = (isoString) => {
  if (!isoString) return 'N/A';
  try {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false
    });
  } catch {
    return isoString;
  }
};

export const getRiskMeta = (level) => {
  const norm = (level || 'unknown').toLowerCase();
  switch (norm) {
    case 'safe':
    case 'benign':
    case 'low':
      return {
        label: 'BENIGN / SAFE',
        color: 'text-emerald-400',
        bg: 'bg-emerald-500/10',
        border: 'border-emerald-500/30',
        badgeClass: 'soc-badge-success',
        dotClass: 'bg-emerald-400',
        scoreRange: '0.0 - 0.25'
      };
    case 'suspicious':
    case 'medium':
      return {
        label: 'SUSPICIOUS',
        color: 'text-amber-400',
        bg: 'bg-amber-500/10',
        border: 'border-amber-500/30',
        badgeClass: 'soc-badge-warning',
        dotClass: 'bg-amber-400',
        scoreRange: '0.25 - 0.70'
      };
    case 'malicious':
    case 'malware':
    case 'phishing':
    case 'defacement':
    case 'high':
    case 'critical':
      return {
        label: 'MALICIOUS',
        color: 'text-rose-400',
        bg: 'bg-rose-500/10',
        border: 'border-rose-500/30',
        badgeClass: 'soc-badge-danger',
        dotClass: 'bg-rose-500 animate-ping',
        scoreRange: '0.70 - 1.00'
      };
    default:
      return {
        label: 'UNVERIFIED',
        color: 'text-slate-400',
        bg: 'bg-slate-500/10',
        border: 'border-slate-500/30',
        badgeClass: 'bg-slate-800 text-slate-300 border border-slate-700',
        dotClass: 'bg-slate-400',
        scoreRange: 'N/A'
      };
  }
};

export const copyToClipboard = async (text) => {
  if (!text) return false;
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (err) {
    console.error('Failed to copy to clipboard', err);
    return false;
  }
};

export const exportToCSV = (filename, rows) => {
  if (!rows || !rows.length) return;
  const separator = ',';
  const keys = Object.keys(rows[0]);
  const csvContent =
    keys.join(separator) +
    '\n' +
    rows
      .map(row => {
        return keys
          .map(k => {
            let cell = row[k] === null || row[k] === undefined ? '' : row[k];
            cell = cell instanceof Date ? cell.toLocaleString() : cell.toString();
            cell = cell.replace(/"/g, '""');
            if (cell.search(/("|,|\n)/g) >= 0) cell = `"${cell}"`;
            return cell;
          })
          .join(separator);
      })
      .join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};
