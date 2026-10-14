import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { BackendProvider } from './context/BackendContext';
import { ScanProvider } from './context/ScanContext';
import AppLayout from './components/layout/AppLayout';

// Page components
import Dashboard from './pages/Dashboard';
import ThreatAnalyzer from './pages/ThreatAnalyzer';
import URLIntelligence from './pages/URLIntelligence';
import MalwareAnalysis from './pages/MalwareAnalysis';
import InfrastructureGraph from './pages/InfrastructureGraph';
import DynamicAnalysis from './pages/DynamicAnalysis';
import ExplainableAI from './pages/ExplainableAI';
import ScanHistory from './pages/ScanHistory';
import SecurityAssistant from './pages/SecurityAssistant';
import Settings from './pages/Settings';
import NotFound from './pages/NotFound';

function App() {
  return (
    <BackendProvider>
      <ScanProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<AppLayout />}>
              <Route index element={<Dashboard />} />
              <Route path="threat-analyzer" element={<ThreatAnalyzer />} />
              <Route path="url-intelligence" element={<URLIntelligence />} />
              <Route path="malware-analysis" element={<MalwareAnalysis />} />
              <Route path="infrastructure-graph" element={<InfrastructureGraph />} />
              <Route path="dynamic-analysis" element={<DynamicAnalysis />} />
              <Route path="explainable-ai" element={<ExplainableAI />} />
              <Route path="scan-history" element={<ScanHistory />} />
              <Route path="security-assistant" element={<SecurityAssistant />} />
              <Route path="settings" element={<Settings />} />
              <Route path="*" element={<NotFound />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ScanProvider>
    </BackendProvider>
  );
}

export default App;
