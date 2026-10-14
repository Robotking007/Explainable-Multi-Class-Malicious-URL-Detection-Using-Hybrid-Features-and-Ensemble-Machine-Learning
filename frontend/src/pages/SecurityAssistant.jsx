import React, { useState, useRef, useEffect } from 'react';
import {
  Bot,
  User,
  Send,
  Trash2,
  Sparkles,
  AlertCircle,
  Shield,
  Layers,
  Terminal,
  RefreshCw,
  Info
} from 'lucide-react';
import chatbotService from '../services/chatbotService';
import { useBackend } from '../context/BackendContext';
import { useScans } from '../context/ScanContext';

export const SecurityAssistant = () => {
  const { isOnline, apiUrl } = useBackend();
  const { activeScan } = useScans();

  const [messages, setMessages] = useState([
    {
      id: 'init-msg',
      role: 'assistant',
      content:
        'Welcome to the CyberX AI Security Assistant workstation. You can query me on Graph Neural Network URL inference, zero-day malware behavioral patterns (CIC-MalMem-2022, BODMAS, MalAPI2019), or ask for interpretation of SHAP/Gradient attribution scores.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (textToSend) => {
    const text = textToSend || inputText;
    if (!text || !text.trim() || loading) return;

    setError(null);
    const userMsg = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text.trim(),
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText('');
    setLoading(true);

    try {
      const res = await chatbotService.sendMessage(
        text.trim(),
        messages,
        activeScan ? { target: activeScan.url || activeScan.fileName, risk: activeScan.riskLevel } : null
      );

      const botMsg = {
        id: `b-${Date.now()}`,
        role: 'assistant',
        content: res.content,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setError(err.message || 'Failed to reach FastAPI assistant endpoint.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessages([
      {
        id: 'reset',
        role: 'assistant',
        content: 'Session cleared. Ready for your cybersecurity or model interpretability query.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
    setError(null);
  };

  const suggestedQuestions = chatbotService.getSuggestedQuestions();

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="soc-badge-cyan text-xs font-mono">LLM & XAI COPILOT</span>
            <span className="text-xs text-slate-400 font-mono">• Interactive SOC Assistant</span>
          </div>
          <h1 className="text-2xl font-bold text-slate-100 font-mono tracking-tight mt-1">
            AI Security Assistant
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl mt-1">
            Dedicated domain assistant connected to your FastAPI backend for interactive threat explanation and GNN graph analysis interpretation.
          </p>
        </div>

        <button
          onClick={handleClear}
          className="soc-btn-secondary text-xs px-3.5 py-2 font-mono flex items-center gap-2"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Clear Dialogue</span>
        </button>
      </div>

      {/* Main Chat Interface Container */}
      <div className="soc-card overflow-hidden h-[620px] flex flex-col">
        {/* Chat Messages Log */}
        <div className="flex-1 p-5 overflow-y-auto space-y-4 font-sans text-xs">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex gap-3 ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {m.role === 'assistant' && (
                <div className="w-8 h-8 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 flex items-center justify-center flex-shrink-0 mt-0.5 shadow-glow-cyan">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`max-w-[75%] rounded-2xl px-4 py-3 leading-relaxed ${
                  m.role === 'user'
                    ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-medium rounded-tr-none'
                    : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none shadow-md'
                }`}
              >
                <p className="whitespace-pre-wrap">{m.content}</p>
                <span
                  className={`block text-[10px] mt-1.5 font-mono text-right ${
                    m.role === 'user' ? 'text-cyan-950/70' : 'text-slate-400'
                  }`}
                >
                  {m.timestamp}
                </span>
              </div>

              {m.role === 'user' && (
                <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex items-center gap-2.5 text-slate-400 text-xs italic py-3">
              <Bot className="w-4 h-4 text-cyan-400 animate-spin" />
              <span className="font-mono">FastAPI Assistant is evaluating prompt context...</span>
            </div>
          )}

          {error && (
            <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-bold font-mono block">Backend Integration Notice:</span>
                <span className="leading-relaxed block mt-0.5">{error}</span>
                <div className="mt-2 text-[11px] text-slate-400 font-mono">
                  Endpoint: <code className="text-cyan-300">{apiUrl}/api/chat</code>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Suggested Prompts Shelf */}
        <div className="px-4 py-2.5 bg-[#080d19] border-t border-slate-800 flex items-center gap-2 overflow-x-auto">
          <span className="text-[11px] font-mono text-slate-400 flex items-center gap-1 flex-shrink-0">
            <Sparkles className="w-3 h-3 text-cyan-400" />
            Quick Prompts:
          </span>
          {suggestedQuestions.map((q, idx) => (
            <button
              key={idx}
              onClick={() => handleSend(q)}
              disabled={loading}
              className="whitespace-nowrap px-3 py-1 rounded-full bg-slate-800/80 hover:bg-slate-700 text-xs text-cyan-300 border border-cyan-500/20 font-mono transition-colors disabled:opacity-40"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="p-4 bg-[#090e1c] border-t border-slate-800 flex gap-3">
          <input
            type="text"
            placeholder="Type your cybersecurity or Explainable AI question..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={loading}
            className="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-3 text-xs text-slate-100 placeholder-slate-500 font-sans focus:outline-none focus:border-cyan-400"
          />
          <button
            type="submit"
            disabled={loading || !inputText.trim()}
            className="soc-btn-primary px-6 py-3 text-xs font-mono"
          >
            <Send className="w-4 h-4" />
            <span>Send Query</span>
          </button>
        </form>
      </div>
    </div>
  );
};

export default SecurityAssistant;
