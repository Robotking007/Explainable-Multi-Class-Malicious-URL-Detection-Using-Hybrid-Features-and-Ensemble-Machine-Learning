import React, { useState, useEffect } from 'react';
import {
  BrainCircuit,
  Search,
  Cpu,
  Layers,
  Sparkles,
  BarChart3,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Info,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  Cell
} from 'recharts';
import explainabilityService from '../services/explainabilityService';
import { useScans } from '../context/ScanContext';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorBanner from '../components/common/ErrorBanner';

export const ExplainableAI = () => {
  const { activeScan } = useScans();
  const [scanQuery, setScanQuery] = useState(activeScan?.url || 'http://bank-secure-update.account-verification.xyz/login.php');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [explanationData, setExplanationData] = useState(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const fetchExplanation = async (target = scanQuery) => {
    setLoading(true);
    setError(null);
    try {
      const data = await explainabilityService.explainUrlPrediction(target);
      setExplanationData(data);
    } catch (err) {
      // Backend explainability endpoint unavailable fallback
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExplanation(scanQuery);
  }, []);

  // Format chart data with positive (red/pushing towards malicious) vs negative (green/pushing towards benign)
  const chartData = (explanationData?.topFeatures || []).map((f) => ({
    name: f.feature.length > 28 ? f.feature.slice(0, 26) + '...' : f.feature,
    fullFeature: f.feature,
    impact: f.impact,
    description: f.description,
    direction: f.direction
  }));

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">XAI ENGINE</span>
            <span className="text-xs text-slate-400 font-mono">• Model Transparency & Attribution</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            Explainable AI (XAI) Dashboard
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Deconstruct black-box deep learning decisions into quantitative SHAP attribution values and neural gradient sensitivities.
          </p>
        </div>
      </div>

      {/* Causation Warning Banner */}
      <div className="p-4 rounded-xl bg-slate-900/80 border border-violet-500/25 flex items-start gap-3 text-xs text-slate-300">
        <Info className="w-5 h-5 text-violet-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <span className="font-semibold text-slate-200 font-mono">Academic XAI Interpretability Disclosure: </span>
          <span>
            Feature attributions reflect mathematical contribution weights (Integrated Gradients & TreeSHAP)
            towards the model's output probability. They do not demonstrate absolute physical causation or replace manual forensic validation.
          </span>
        </div>
      </div>

      {/* Query Bar */}
      <div className="soc-card p-4 flex flex-col sm:flex-row gap-3">
        <input
          type="text"
          value={scanQuery}
          onChange={(e) => setScanQuery(e.target.value)}
          placeholder="Target URL or Scan ID..."
          className="flex-1 bg-[#080d19] border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-400"
        />
        <button
          onClick={() => fetchExplanation(scanQuery)}
          disabled={loading}
          className="soc-btn-primary px-6 py-2.5 text-xs font-mono"
        >
          <Cpu className="w-4 h-4" />
          <span>Compute Attributions</span>
        </button>
      </div>

      {loading && (
        <LoadingSpinner
          message="Computing Integrated Gradients & Shapley Kernel"
          subtext="Evaluating baseline interpolations across URL token embeddings and graph adjacency..."
        />
      )}

      {error && (
        <ErrorBanner
          error={error}
          title="XAI Computation Notice"
          onRetry={() => fetchExplanation(scanQuery)}
        />
      )}

      {explanationData && !loading && (
        <div className="space-y-6">
          {/* Summary Card */}
          <div className="soc-card p-6 border-l-4 border-l-violet-500 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
              <div>
                <span className="text-[10px] font-mono text-slate-400">MODEL ARCHITECTURE</span>
                <h3 className="text-base font-bold font-mono text-violet-300">{explanationData.modelType}</h3>
                <span className="text-xs text-slate-400 font-mono">Method: {explanationData.attributionMethod}</span>
              </div>
              <div className="text-right font-mono text-xs">
                <span className="text-slate-400 block text-[10px]">PREDICTED VERDICT</span>
                <span className="text-rose-400 font-bold text-sm">{explanationData.prediction}</span>
              </div>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed font-sans bg-slate-900/60 p-4 rounded-xl border border-slate-800">
              <span className="font-semibold text-cyan-300 font-mono">Explanation Summary: </span>
              {explanationData.summary}
            </p>
          </div>

          {/* Feature Importance Bar Chart */}
          <div className="soc-card p-6 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-bold font-mono text-slate-100">
                  SHAP Feature Attribution Weights
                </h3>
                <p className="text-[11px] text-slate-400">
                  Positive values push verdict toward Malicious (Red); negative toward Benign (Green)
                </p>
              </div>
              <BarChart3 className="w-4 h-4 text-cyan-400" />
            </div>

            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 30, left: 140, bottom: 5 }}>
                  <XAxis type="number" stroke="#64748b" fontSize={11} domain={[-0.5, 0.5]} />
                  <YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={11} width={130} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#090e1c',
                      borderColor: '#1e293b',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                    formatter={(val) => [`Impact: ${val}`, 'Attribution']}
                  />
                  <ReferenceLine x={0} stroke="#475569" strokeWidth={1.5} />
                  <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
                    {chartData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={entry.impact >= 0 ? '#ef4444' : '#10b981'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Feature Breakdown Table */}
          <div className="soc-card p-6 space-y-4">
            <h3 className="text-sm font-bold font-mono text-slate-100">
              Ranked Feature Descriptions & Reasoning
            </h3>

            <div className="space-y-3 font-mono text-xs">
              {explanationData.topFeatures?.map((f, i) => (
                <div
                  key={i}
                  className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-3"
                >
                  <div className="space-y-1">
                    <span className="font-bold text-slate-200">{f.feature}</span>
                    <p className="text-[11px] text-slate-400 font-sans">{f.description}</p>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span
                      className={`px-2 py-1 rounded text-xs font-bold ${
                        f.impact >= 0
                          ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                          : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                      }`}
                    >
                      {f.impact >= 0 ? `+${f.impact}` : f.impact}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Technical Details Accordion */}
          <div className="soc-card p-5 space-y-3">
            <button
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              className="w-full flex items-center justify-between text-xs font-mono font-bold text-slate-300"
            >
              <span>Inspect Neural Gradient Layers & Convergence Norms</span>
              {showTechnicalDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>

            {showTechnicalDetails && (
              <div className="pt-3 border-t border-slate-800 space-y-2 font-mono text-xs">
                {explanationData.neuralGradients?.map((g, idx) => (
                  <div key={idx} className="flex justify-between p-2 rounded bg-slate-900 border border-slate-800 text-[11px]">
                    <span className="text-slate-400">{g.layer}</span>
                    <span className="text-cyan-400">Mean Grad: {g.meanGradient} | Activation: {g.activationNorm}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ExplainableAI;
