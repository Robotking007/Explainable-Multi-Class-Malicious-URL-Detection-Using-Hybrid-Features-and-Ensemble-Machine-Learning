import React, { useState, useEffect, useRef } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Shield,
  Globe,
  Binary,
  Search,
  UploadCloud,
  FileCode,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  StopCircle,
  Sparkles,
  ExternalLink,
  Copy,
  Check,
  RefreshCw,
  Cpu,
  Layers,
  ArrowRight,
  Info
} from 'lucide-react';
import threatService from '../services/threatService';
import malwareService, { MALWARE_MODULES } from '../services/malwareService';
import { useBackend } from '../context/BackendContext';
import { useScans } from '../context/ScanContext';
import StatusBadge from '../components/common/StatusBadge';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorBanner from '../components/common/ErrorBanner';
import { copyToClipboard, formatTimestamp } from '../utils/helpers';

export const ThreatAnalyzer = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { isOnline, apiUrl } = useBackend();
  const { addScan, setActiveScan } = useScans();

  // Mode: 'url' | 'malware'
  const [activeTab, setActiveTab] = useState('url');

  // URL Analysis State
  const [urlInput, setUrlInput] = useState('');
  const [urlLoading, setUrlLoading] = useState(false);
  const [urlError, setUrlError] = useState(null);
  const [urlResult, setUrlResult] = useState(null);
  const [copied, setCopied] = useState(false);
  const abortControllerRef = useRef(null);

  // Malware Analysis State
  const [selectedModule, setSelectedModule] = useState(MALWARE_MODULES[0].id);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [malwareLoading, setMalwareLoading] = useState(false);
  const [malwareError, setMalwareError] = useState(null);
  const [malwareResult, setMalwareResult] = useState(null);

  // Handle URL query parameter pre-population
  useEffect(() => {
    const query = searchParams.get('query');
    if (query) {
      setUrlInput(query);
      setActiveTab('url');
    }
  }, [searchParams]);

  // URL Sample presets
  const sampleUrls = [
    { label: 'Phishing Sample', url: 'http://secure-paypal-login-verify.com/account/login.php' },
    { label: 'Malware Dropper', url: 'http://185.220.101.5/payload.bin' },
    { label: 'Defacement Test', url: 'http://hacked-by-darkteam.org/index.html' },
    { label: 'Benign University Domain', url: 'https://www.stanford.edu/research/cybersecurity' }
  ];

  // ================= URL ANALYSIS HANDLERS =================
  const handleAnalyzeUrl = async (targetUrl = urlInput) => {
    const urlToTest = targetUrl || urlInput;
    if (!urlToTest || !urlToTest.trim()) {
      setUrlError('Please enter a target URL to analyze.');
      return;
    }

    // Basic format validation
    let sanitized = urlToTest.trim();
    if (!/^https?:\/\//i.test(sanitized)) {
      sanitized = `http://${sanitized}`;
    }

    setUrlError(null);
    setUrlResult(null);
    setUrlLoading(true);

    // Setup abort controller for cancelability
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    try {
      const result = await threatService.analyzeUrl(sanitized, abortControllerRef.current.signal);
      setUrlResult(result);
      addScan({
        id: result.id,
        type: 'URL',
        url: result.url,
        classification: result.classification,
        riskLevel: result.riskLevel,
        confidence: result.confidence,
        timestamp: result.timestamp,
        features: result.features
      });
    } catch (err) {
      if (err.name === 'CanceledError' || err.message?.includes('canceled')) {
        setUrlError('URL analysis was canceled by user.');
      } else {
        setUrlError(err);
      }
    } finally {
      setUrlLoading(false);
    }
  };

  const handleCancelUrl = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setUrlLoading(false);
  };

  // ================= MALWARE ANALYSIS HANDLERS =================
  const handleFileDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelected = (file) => {
    const currentMod = MALWARE_MODULES.find((m) => m.id === selectedModule);
    if (!file) return;

    // Size validation
    if (file.size > currentMod.maxSizeMb * 1024 * 1024) {
      setMalwareError(`File exceeds maximum size of ${currentMod.maxSizeMb}MB for this engine.`);
      return;
    }

    setUploadedFile(file);
    setMalwareError(null);
    setMalwareResult(null);
  };

  const handleAnalyzeMalware = async () => {
    if (!uploadedFile) {
      setMalwareError('Please upload a file before initiating analysis.');
      return;
    }

    setMalwareLoading(true);
    setMalwareError(null);
    setMalwareResult(null);
    setUploadProgress(0);

    try {
      const result = await malwareService.analyzeMalwareFile(
        uploadedFile,
        selectedModule,
        (progress) => setUploadProgress(progress)
      );
      setMalwareResult(result);
      addScan({
        id: result.id,
        type: 'MALWARE',
        fileName: result.fileName,
        moduleName: result.module?.name,
        classification: result.predictedFamily,
        riskLevel: result.riskLevel,
        confidence: result.confidence,
        timestamp: result.timestamp
      });
    } catch (err) {
      setMalwareError(err);
    } finally {
      setMalwareLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">WORKSPACE</span>
            <span className="text-xs text-slate-400 font-mono">• Unified Classifier</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            Threat Analyzer
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Analyze untrusted URLs & zero-day malware binaries using multi-modal Graph Neural Networks with live model inference.
          </p>
        </div>

        {/* Tab switch controls */}
        <div className="flex p-1 bg-slate-900/90 border border-slate-800 rounded-xl">
          <button
            onClick={() => setActiveTab('url')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono font-medium transition-all ${
              activeTab === 'url'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-glow-cyan'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Globe className="w-4 h-4" />
            <span>Malicious URL Analysis</span>
          </button>
          <button
            onClick={() => setActiveTab('malware')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-mono font-medium transition-all ${
              activeTab === 'malware'
                ? 'bg-violet-500/20 text-violet-300 border border-violet-500/40 shadow-glow-violet'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Binary className="w-4 h-4" />
            <span>Malware Binary Analysis</span>
          </button>
        </div>
      </div>

      {/* Safety Notice Banner */}
      <div className="p-3.5 rounded-xl bg-slate-900/80 border border-cyan-500/20 flex items-start gap-3 text-xs text-slate-300">
        <Info className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-0.5 leading-relaxed">
          <span className="font-semibold text-slate-200 font-mono">SOC Operational Safety Protocol: </span>
          <span>
            All submitted payloads and URLs are evaluated remotely in isolated sandboxes or neural inference pipelines.
            Uploaded binaries are never executed in the client browser. Model predictions reflect probability scores rather than absolute physical safety guarantees.
          </span>
        </div>
      </div>

      {/* ============================================================== */}
      {/* MODE A: MALICIOUS URL ANALYSIS                                   */}
      {/* ============================================================== */}
      {activeTab === 'url' && (
        <div className="space-y-6">
          {/* URL Input Form Card */}
          <div className="soc-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <label htmlFor="url-input" className="text-xs font-mono font-semibold text-slate-200 flex items-center gap-2">
                <Globe className="w-4 h-4 text-cyan-400" />
                <span>Target URL for Neural & Infrastructure Analysis</span>
              </label>
              {urlLoading && (
                <button
                  onClick={handleCancelUrl}
                  className="flex items-center gap-1.5 text-xs text-rose-400 hover:text-rose-300 font-mono"
                >
                  <StopCircle className="w-3.5 h-3.5" />
                  <span>Cancel Request</span>
                </button>
              )}
            </div>

            <div className="flex flex-col sm:flex-row gap-3">
              <div className="relative flex-1">
                <input
                  id="url-input"
                  type="text"
                  placeholder="https://example.com/login-verification or IP address..."
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleAnalyzeUrl()}
                  disabled={urlLoading}
                  className="w-full bg-[#080d19] border border-slate-700/80 rounded-xl px-4 py-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-cyan-400 focus:ring-1 focus:ring-cyan-400/30 font-mono transition-all"
                />
                {urlInput && (
                  <button
                    onClick={() => setUrlInput('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 text-xs"
                    title="Clear Input"
                  >
                    Clear
                  </button>
                )}
              </div>

              <button
                onClick={() => handleAnalyzeUrl()}
                disabled={urlLoading || !urlInput.trim()}
                className="soc-btn-primary px-6 py-3 text-xs whitespace-nowrap shadow-glow-cyan"
              >
                {urlLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <>
                    <Shield className="w-4 h-4" />
                    <span>Analyze URL</span>
                  </>
                )}
              </button>
            </div>

            {/* Presets / Sample URLs */}
            <div className="pt-2 border-t border-slate-800/80 flex flex-wrap items-center gap-2">
              <span className="text-[11px] text-slate-500 font-mono flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-cyan-400" />
                Sample Presets:
              </span>
              {sampleUrls.map((s, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setUrlInput(s.url);
                    handleAnalyzeUrl(s.url);
                  }}
                  className="px-2.5 py-1 rounded bg-slate-900/90 hover:bg-slate-800 text-slate-300 hover:text-cyan-300 border border-slate-800 text-[11px] font-mono transition-colors"
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* Loading Indicator */}
          {urlLoading && (
            <LoadingSpinner
              message="Processing URL with URLInfraGNN & Static Parsers"
              subtext="Extracting lexical entropy, DNS bindings, redirect chains, and ASN node features..."
              step="GNN Embedding Layer 2"
            />
          )}

          {/* Error Banner */}
          {urlError && (
            <ErrorBanner
              error={urlError}
              title="URL Analysis Failed"
              onRetry={() => handleAnalyzeUrl()}
            />
          )}

          {/* Results Panel */}
          {urlResult && !urlLoading && (
            <div className="space-y-6">
              {/* Verdict Overview Card */}
              <div className="soc-card p-6 border-l-4 border-l-cyan-400 space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
                  <div className="space-y-1">
                    <span className="text-[11px] font-mono text-slate-400">TARGET ARTIFACT</span>
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm md:text-base font-bold font-mono text-cyan-300 break-all">
                        {urlResult.url}
                      </h3>
                      <button
                        onClick={async () => {
                          const ok = await copyToClipboard(urlResult.url);
                          if (ok) {
                            setCopied(true);
                            setTimeout(() => setCopied(false), 2000);
                          }
                        }}
                        className="p-1 rounded text-slate-400 hover:text-slate-200 transition-colors"
                        title="Copy URL"
                      >
                        {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <StatusBadge status={urlResult.riskLevel} size="lg" pulse={urlResult.riskLevel?.toLowerCase() === 'malicious'} />
                  </div>
                </div>

                {/* Score & Category Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Classification</span>
                    <p className="text-lg font-bold font-mono text-slate-100 mt-1 capitalize">
                      {urlResult.classification}
                    </p>
                    <span className="text-[10px] text-slate-400">Predicted Class Category</span>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Confidence Score</span>
                    <p className="text-lg font-bold font-mono text-cyan-400 mt-1">
                      {urlResult.confidence !== null
                        ? `${(urlResult.confidence * (urlResult.confidence <= 1 ? 100 : 1)).toFixed(2)}%`
                        : 'Unreported by Backend'}
                    </p>
                    <span className="text-[10px] text-slate-400">Softmax Model Probability</span>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Assigned Risk Level</span>
                    <p className="text-lg font-bold font-mono text-slate-100 mt-1">
                      {urlResult.riskLevel}
                    </p>
                    <span className="text-[10px] text-slate-400">SOC Severity Matrix</span>
                  </div>

                  <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
                    <span className="text-[10px] font-mono text-slate-400 uppercase">Scan Identifier</span>
                    <p className="text-sm font-bold font-mono text-slate-300 mt-1 truncate">
                      {urlResult.id}
                    </p>
                    <span className="text-[10px] text-slate-400">{formatTimestamp(urlResult.timestamp)}</span>
                  </div>
                </div>

                {/* Extracted Features & Explainability Summary */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Extracted URL Features */}
                  <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3 font-mono text-xs">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                      <span className="font-semibold text-slate-200">Lexical & Structural Features</span>
                      <Layers className="w-3.5 h-3.5 text-cyan-400" />
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-400 block">URL Length:</span>
                        <span className="text-slate-200 font-bold">{urlResult.features?.url_length || urlResult.url.length} chars</span>
                      </div>
                      <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-400 block">Contains IP in Host:</span>
                        <span className="text-slate-200 font-bold">{urlResult.features?.has_ip ? 'Yes (Flagged)' : 'No'}</span>
                      </div>
                      <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-400 block">At-Symbol (@):</span>
                        <span className="text-slate-200 font-bold">{urlResult.features?.has_at_symbol ? 'Present' : 'Absent'}</span>
                      </div>
                      <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
                        <span className="text-slate-400 block">HTTPS Protocol:</span>
                        <span className="text-slate-200 font-bold">{urlResult.features?.is_https ? 'Secure (TLS)' : 'Unencrypted'}</span>
                      </div>
                    </div>
                  </div>

                  {/* Explainability Summary & Actions */}
                  <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3 font-mono text-xs flex flex-col justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                        <span className="font-semibold text-slate-200">Explainable AI Summary</span>
                        <Sparkles className="w-3.5 h-3.5 text-violet-400" />
                      </div>
                      <p className="text-[11px] text-slate-300 leading-relaxed font-sans">
                        {urlResult.explainabilitySummary ||
                          'GNN model evaluated spatial clustering across IP, DNS, and autonomous system nodes to produce this classification.'}
                      </p>
                    </div>

                    <div className="flex flex-wrap gap-2 pt-2">
                      <button
                        onClick={() => navigate('/explainable-ai')}
                        className="soc-btn-primary text-xs px-3.5 py-2 flex items-center gap-1.5"
                      >
                        <Cpu className="w-3.5 h-3.5" />
                        <span>Inspect SHAP & Gradients</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => navigate('/infrastructure-graph')}
                        className="soc-btn-secondary text-xs px-3.5 py-2 flex items-center gap-1.5"
                      >
                        <ExternalLink className="w-3.5 h-3.5 text-cyan-400" />
                        <span>View Infrastructure Graph</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ============================================================== */}
      {/* MODE B: MALWARE BINARY ANALYSIS                                  */}
      {/* ============================================================== */}
      {activeTab === 'malware' && (
        <div className="space-y-6">
          {/* Module Selector */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {MALWARE_MODULES.map((mod) => (
              <div
                key={mod.id}
                onClick={() => setSelectedModule(mod.id)}
                className={`cursor-pointer p-4 rounded-xl border transition-all ${
                  selectedModule === mod.id
                    ? 'bg-violet-950/30 border-violet-500 shadow-glow-violet'
                    : 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-100 font-mono">{mod.shortName}</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-violet-400 border border-slate-700">
                    {mod.dataset}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">{mod.description}</p>
                <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] font-mono text-slate-500">
                  <span>Formats: {mod.acceptedExtensions.join(', ')}</span>
                  <span>Max: {mod.maxSizeMb}MB</span>
                </div>
              </div>
            ))}
          </div>

          {/* Drag & Drop Upload Zone */}
          <div className="soc-card p-6 space-y-4">
            <div
              onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
              className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
                dragActive
                  ? 'border-violet-500 bg-violet-500/10'
                  : 'border-slate-800 bg-[#080d19] hover:border-slate-700'
              }`}
            >
              <UploadCloud className="w-12 h-12 text-violet-400 mx-auto mb-3" />
              <h3 className="text-sm font-semibold text-slate-200">
                Drag and drop your malware dump or binary sample here
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Accepted types for selected module: {MALWARE_MODULES.find((m) => m.id === selectedModule)?.acceptedExtensions.join(', ')}
              </p>

              <div className="mt-4">
                <input
                  type="file"
                  id="malware-file-input"
                  className="hidden"
                  onChange={(e) => e.target.files && handleFileSelected(e.target.files[0])}
                />
                <label
                  htmlFor="malware-file-input"
                  className="soc-btn-secondary text-xs px-4 py-2 cursor-pointer inline-flex items-center gap-2"
                >
                  <FileCode className="w-4 h-4 text-cyan-400" />
                  <span>Browse Local File</span>
                </label>
              </div>

              {uploadedFile && (
                <div className="mt-4 p-3 rounded-lg bg-slate-900 border border-slate-800 inline-flex items-center gap-3 text-xs font-mono text-slate-300">
                  <FileCode className="w-4 h-4 text-violet-400" />
                  <span>{uploadedFile.name}</span>
                  <span className="text-slate-500">({(uploadedFile.size / 1024).toFixed(1)} KB)</span>
                  <button
                    onClick={() => setUploadedFile(null)}
                    className="text-slate-500 hover:text-rose-400 text-xs ml-2"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>

            {uploadedFile && (
              <div className="flex justify-end">
                <button
                  onClick={handleAnalyzeMalware}
                  disabled={malwareLoading}
                  className="soc-btn-primary px-6 py-2.5 text-xs font-mono bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 shadow-glow-violet"
                >
                  {malwareLoading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Running Neural Classifier...</span>
                    </>
                  ) : (
                    <>
                      <Binary className="w-4 h-4" />
                      <span>Start Malware Inference</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Upload Progress */}
          {malwareLoading && (
            <LoadingSpinner
              message={`Running ${MALWARE_MODULES.find((m) => m.id === selectedModule)?.name}`}
              subtext="Parsing bytecode headers, VAD trees, and sequential API transitions..."
              step="Forward Pass"
            />
          )}

          {/* Error Banner */}
          {malwareError && (
            <ErrorBanner
              error={malwareError}
              title="Malware Analysis Pipeline Error"
              onRetry={handleAnalyzeMalware}
            />
          )}

          {/* Malware Result Panel */}
          {malwareResult && !malwareLoading && (
            <div className="soc-card p-6 border-l-4 border-l-violet-500 space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
                <div>
                  <span className="text-[11px] font-mono text-slate-400">ANALYZED SAMPLE</span>
                  <h3 className="text-base font-bold font-mono text-violet-300">{malwareResult.fileName}</h3>
                  <span className="text-xs text-slate-400 font-mono">Module: {malwareResult.module?.name}</span>
                </div>
                <StatusBadge status={malwareResult.riskLevel} size="lg" />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono text-xs">
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 uppercase text-[10px] block">Predicted Family</span>
                  <span className="text-base font-bold text-slate-100 block mt-1">{malwareResult.predictedFamily}</span>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 uppercase text-[10px] block">Confidence</span>
                  <span className="text-base font-bold text-violet-400 block mt-1">
                    {malwareResult.confidence !== null
                      ? `${(malwareResult.confidence * (malwareResult.confidence <= 1 ? 100 : 1)).toFixed(1)}%`
                      : 'N/A'}
                  </span>
                </div>
                <div className="p-4 rounded-xl bg-slate-900 border border-slate-800">
                  <span className="text-slate-400 uppercase text-[10px] block">Pipeline Hash</span>
                  <span className="text-xs text-slate-300 block mt-1 truncate">{malwareResult.sha256}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ThreatAnalyzer;
