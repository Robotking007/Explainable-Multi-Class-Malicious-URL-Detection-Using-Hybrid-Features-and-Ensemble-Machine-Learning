import React, { createContext, useContext, useState, useEffect } from 'react';
import historyService from '../services/historyService';

const ScanContext = createContext(null);

export const ScanProvider = ({ children }) => {
  const [history, setHistory] = useState([]);
  const [activeScan, setActiveScan] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historySource, setHistorySource] = useState('local');

  const loadHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await historyService.getScanHistory();
      setHistory(res.scans);
      setHistorySource(res.source);
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const addScan = async (scan) => {
    setActiveScan(scan);
    await historyService.saveScan(scan);
    setHistory((prev) => [scan, ...prev.filter((s) => s.id !== scan.id)]);
  };

  const clearAllHistory = () => {
    historyService.clearHistory();
    setHistory([]);
    setActiveScan(null);
  };

  return (
    <ScanContext.Provider
      value={{
        history,
        activeScan,
        setActiveScan,
        addScan,
        loadingHistory,
        historySource,
        refreshHistory: loadHistory,
        clearAllHistory,
      }}
    >
      {children}
    </ScanContext.Provider>
  );
};

export const useScans = () => {
  const context = useContext(ScanContext);
  if (!context) {
    throw new Error('useScans must be used within a ScanProvider');
  }
  return context;
};

export default ScanContext;
