import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { checkBackendHealth, setApiBaseUrl, getApiBaseUrl } from '../services/api';

const BackendContext = createContext(null);

export const BackendProvider = ({ children }) => {
  const [status, setStatus] = useState('checking'); // 'checking' | 'online' | 'offline'
  const [latency, setLatency] = useState(null);
  const [lastChecked, setLastChecked] = useState(null);
  const [backendMeta, setBackendMeta] = useState(null);
  const [apiUrl, setApiUrlState] = useState(getApiBaseUrl());
  const [moduleStatus, setModuleStatus] = useState({
    urlGNN: { name: 'URLInfraGNN Classifier', status: 'ready' },
    malmem: { name: 'CIC-MalMem-2022 Memory Forensics', status: 'ready' },
    bodmas: { name: 'BODMAS & EMBER Static PE Analyzer', status: 'ready' },
    malapi: { name: 'MalAPI2019 API Sequence GNN', status: 'ready' },
    xaiEngine: { name: 'Integrated Gradients & SHAP Attributor', status: 'ready' },
    sandbox: { name: 'Dynamic Browser Sandbox Isolation', status: 'controlled' }
  });

  const checkConnection = useCallback(async () => {
    setStatus('checking');
    try {
      const res = await checkBackendHealth();
      setLastChecked(new Date());
      if (res.online) {
        setStatus('online');
        setLatency(res.latencyMs);
        setBackendMeta(res.data || { version: 'FastAPI Backend', endpoint: res.endpointUsed });
      } else {
        setStatus('offline');
        setLatency(null);
        setBackendMeta(null);
      }
    } catch {
      setStatus('offline');
      setLatency(null);
      setBackendMeta(null);
    }
  }, []);

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, 30000);
    return () => clearInterval(interval);
  }, [checkConnection]);

  const updateApiUrl = (newUrl) => {
    setApiBaseUrl(newUrl);
    setApiUrlState(newUrl);
    checkConnection();
  };

  return (
    <BackendContext.Provider
      value={{
        status,
        isOnline: status === 'online',
        latency,
        lastChecked,
        backendMeta,
        apiUrl,
        updateApiUrl,
        refreshStatus: checkConnection,
        moduleStatus,
        setModuleStatus
      }}
    >
      {children}
    </BackendContext.Provider>
  );
};

export const useBackend = () => {
  const context = useContext(BackendContext);
  if (!context) {
    throw new Error('useBackend must be used within a BackendProvider');
  }
  return context;
};

export default BackendContext;
