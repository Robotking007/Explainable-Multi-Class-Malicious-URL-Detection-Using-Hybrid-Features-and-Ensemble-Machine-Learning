import React, { useState, useEffect, useCallback, useMemo } from 'react';
import ReactFlow, {
  Controls,
  Background,
  applyNodeChanges,
  applyEdgeChanges,
  MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import {
  GitFork,
  Search,
  Maximize2,
  Info,
  Server,
  Globe,
  Layers,
  ShieldAlert,
  ShieldCheck,
  RefreshCw,
  Cpu,
  Radio,
  Sparkles
} from 'lucide-react';
import graphService from '../services/graphService';
import LoadingSpinner from '../components/common/LoadingSpinner';
import ErrorBanner from '../components/common/ErrorBanner';

// Custom node rendering for different SOC infrastructure entities
const CustomNode = ({ data, selected }) => {
  const isUrl = data.category?.includes('URL');
  const isDomain = data.category?.includes('Domain');
  const isIp = data.category?.includes('IP');
  const isAsn = data.category?.includes('Autonomous');

  const borderClass = isUrl
    ? 'border-cyan-400 bg-cyan-950/40 text-cyan-200'
    : isDomain
    ? 'border-violet-400 bg-violet-950/40 text-violet-200'
    : isIp
    ? 'border-rose-400 bg-rose-950/40 text-rose-200'
    : 'border-amber-400 bg-amber-950/40 text-amber-200';

  return (
    <div
      className={`px-3 py-2.5 rounded-xl border-2 shadow-lg backdrop-blur-md font-mono text-xs min-w-[170px] max-w-[240px] transition-all ${borderClass} ${
        selected ? 'ring-2 ring-white scale-105' : ''
      }`}
    >
      <div className="flex items-center justify-between gap-1 text-[9px] uppercase tracking-wider opacity-80 mb-1">
        <span>{data.category || 'Node'}</span>
        {data.risk && (
          <span className="px-1 py-0.2 rounded bg-black/60 font-bold">{data.risk}</span>
        )}
      </div>
      <div className="font-bold truncate text-[11px] text-slate-100">{data.label}</div>
    </div>
  );
};

export const InfrastructureGraph = () => {
  const [targetQuery, setTargetQuery] = useState('http://bank-secure-update.account-verification.xyz/login.php');
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDemoMode, setIsDemoMode] = useState(false);

  const nodeTypes = useMemo(() => ({
    urlNode: CustomNode,
    domainNode: CustomNode,
    ipNode: CustomNode,
    asnNode: CustomNode,
    certNode: CustomNode
  }), []);

  const onNodesChange = useCallback(
    (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange = useCallback(
    (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  const onNodeClick = useCallback((_, node) => {
    setSelectedEntity(node);
  }, []);

  const loadGraph = async (target = targetQuery, forceDemo = false) => {
    setLoading(true);
    setError(null);
    setSelectedEntity(null);

    if (forceDemo) {
      const demo = graphService.getDemoGraph(target);
      setNodes(demo.nodes);
      setEdges(demo.edges);
      setIsDemoMode(true);
      setLoading(false);
      return;
    }

    try {
      const res = await graphService.getInfrastructureGraph(target);
      setNodes(res.nodes);
      setEdges(res.edges);
      setIsDemoMode(false);
    } catch (err) {
      // Backend graph endpoint was not available: fallback smoothly to clearly-marked DEMO mode as specified
      const demo = graphService.getDemoGraph(target);
      setNodes(demo.nodes);
      setEdges(demo.edges);
      setIsDemoMode(true);
      setError('Live URLInfraGNN backend endpoint was not available. Activated clearly labeled Demonstration Topology Mode.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadGraph(targetQuery);
  }, []);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">GNN TOPOLOGY</span>
            <span className="text-xs text-slate-400 font-mono">• URLInfraGNN Spatial Graph</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            Infrastructure Graph Visualization
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Multi-relational graph network mapping dependencies between URLs, Host Domains, BGP Autonomous Systems, and IP endpoints.
          </p>
        </div>

        {/* Demo Mode Indicator */}
        <div className="flex items-center gap-2">
          {isDemoMode && (
            <span className="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/30 text-xs font-mono flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" />
              <span>DEMONSTRATION TOPOLOGY MODE</span>
            </span>
          )}
        </div>
      </div>

      {/* Query Bar */}
      <div className="soc-card p-4 flex flex-col sm:flex-row gap-3">
        <input
          type="text"
          value={targetQuery}
          onChange={(e) => setTargetQuery(e.target.value)}
          placeholder="Target URL or Domain..."
          className="flex-1 bg-[#080d19] border border-slate-700/80 rounded-xl px-4 py-2.5 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-400"
        />
        <button
          onClick={() => loadGraph(targetQuery, false)}
          disabled={loading}
          className="soc-btn-primary px-5 py-2.5 text-xs font-mono"
        >
          <Search className="w-4 h-4" />
          <span>Fetch Topology</span>
        </button>
        <button
          onClick={() => loadGraph(targetQuery, true)}
          disabled={loading}
          className="soc-btn-secondary px-4 py-2.5 text-xs font-mono"
        >
          <span>Load Demo Schema</span>
        </button>
      </div>

      {/* Notice Banner */}
      {isDemoMode && (
        <div className="p-3.5 rounded-xl bg-amber-950/20 border border-amber-500/25 flex items-start gap-3 text-xs text-amber-200">
          <Info className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-bold font-mono">Academic Demonstration Notice: </span>
            <span>
              This graph displays synthetic topology structure adhering to the URLInfraGNN relational schema (URL → Domain → IP → ASN).
              No fabricated intelligence is represented as live detection.
            </span>
          </div>
        </div>
      )}

      {loading && (
        <LoadingSpinner
          message="Constructing GNN Infrastructure Adjacency Matrix"
          subtext="Mapping edges across DNS records, ASN routing, and SSL certificates..."
        />
      )}

      {/* Graph Canvas & Entity Details Drawer */}
      <div className="soc-card overflow-hidden h-[620px] flex flex-col lg:flex-row relative">
        {/* Main Flow Canvas */}
        <div className="flex-1 h-full relative">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
            attributionPosition="bottom-left"
          >
            <Background color="#1e293b" gap={20} size={1} />
            <Controls className="bg-slate-900 border border-slate-700 text-slate-200" />
          </ReactFlow>
        </div>

        {/* Selected Entity Inspector Panel */}
        <div className="w-full lg:w-80 border-t lg:border-t-0 lg:border-l border-slate-800 bg-[#090d1a] p-5 flex flex-col justify-between font-mono text-xs overflow-y-auto">
          {selectedEntity ? (
            <div className="space-y-4">
              <div className="pb-3 border-b border-slate-800">
                <span className="text-[10px] text-slate-400 uppercase">Entity Inspection</span>
                <h3 className="text-sm font-bold text-cyan-300 break-all mt-1">
                  {selectedEntity.data?.label}
                </h3>
                <span className="soc-badge-cyan text-[10px] mt-2">
                  {selectedEntity.data?.category}
                </span>
              </div>

              <div className="space-y-2 text-[11px]">
                {Object.entries(selectedEntity.data || {}).map(([k, v]) => {
                  if (k === 'label' || k === 'category') return null;
                  return (
                    <div key={k} className="p-2.5 rounded bg-slate-900 border border-slate-800">
                      <span className="text-slate-400 block uppercase text-[10px]">{k}:</span>
                      <span className="text-slate-200 font-semibold">{String(v)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center text-center p-6 text-slate-500 my-auto">
              <GitFork className="w-8 h-8 text-slate-600 mb-2" />
              <p className="text-xs">Select any node in the graph to inspect spatial attributes & GNN embeddings.</p>
            </div>
          )}

          <div className="pt-4 border-t border-slate-800 text-[10px] text-slate-500">
            URLInfraGNN Relational Graph Engine
          </div>
        </div>
      </div>
    </div>
  );
};

export default InfrastructureGraph;
