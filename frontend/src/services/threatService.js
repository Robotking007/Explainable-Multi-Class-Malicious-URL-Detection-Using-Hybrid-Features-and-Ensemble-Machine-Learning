import apiClient from './api';

/**
 * Service for URL Threat Analysis
 */
export const threatService = {
  /**
   * Analyze a suspicious URL
   * @param {string} url - Target URL to analyze
   * @param {AbortSignal} signal - Optional abort signal to cancel analysis
   */
  async analyzeUrl(url, signal) {
    if (!url || typeof url !== 'string') {
      throw new Error('Please provide a valid URL string.');
    }

    // Attempt matching standard REST endpoints for URL prediction
    const candidateEndpoints = [
      '/api/analyze-url',
      '/api/v1/analyze/url',
      '/api/url/predict',
      '/predict/url',
      '/analyze'
    ];

    let lastError = null;
    for (const endpoint of candidateEndpoints) {
      try {
        const response = await apiClient.post(
          endpoint,
          { url },
          { signal }
        );
        return normalizeUrlAnalysisResponse(response.data, url);
      } catch (err) {
        if (err.name === 'CanceledError' || err.message?.includes('canceled')) {
          throw err;
        }
        // If route not found (404), try next candidate
        if (err.status === 404) {
          lastError = err;
          continue;
        }
        // If server responded with a real error, propagate it
        throw err;
      }
    }

    // If none matched, report clear unavailable contract
    throw new Error(
      `URL Analysis endpoint is currently unavailable on backend. Please configure one of: ${candidateEndpoints.join(', ')}`
    );
  },

  /**
   * Get supported classification categories
   */
  getCategories() {
    return [
      { id: 'benign', label: 'Benign / Legitimate', color: 'emerald' },
      { id: 'phishing', label: 'Phishing Attempt', color: 'amber' },
      { id: 'defacement', label: 'Web Defacement', color: 'purple' },
      { id: 'malware', label: 'Malware Distribution', color: 'rose' }
    ];
  }
};

/**
 * Normalizes backend response format into a uniform UI object
 */
function normalizeUrlAnalysisResponse(data, originalUrl) {
  if (!data) return null;

  // Handle various potential backend schema response names
  const classification =
    data.classification ||
    data.category ||
    data.prediction ||
    data.predicted_class ||
    data.label ||
    'Unknown';

  const confidence =
    data.confidence !== undefined
      ? data.confidence
      : data.probability !== undefined
      ? data.probability
      : data.score !== undefined
      ? data.score
      : null;

  const riskLevel =
    data.risk_level ||
    data.risk ||
    (classification.toLowerCase() === 'benign'
      ? 'Safe'
      : classification.toLowerCase() === 'phishing'
      ? 'Suspicious'
      : 'Malicious');

  return {
    id: data.scan_id || data.id || `scan-${Date.now().toString(36)}`,
    url: data.url || originalUrl,
    timestamp: data.timestamp || new Date().toISOString(),
    classification: classification,
    riskLevel: riskLevel,
    confidence: confidence !== null ? Number(confidence) : null,
    riskScore: data.risk_score || (confidence ? Math.round(confidence * 100) : null),
    features: data.features || data.url_features || {
      url_length: originalUrl.length,
      has_ip: /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/.test(originalUrl),
      has_at_symbol: originalUrl.includes('@'),
      subdomain_count: (originalUrl.split('/')[2] || '').split('.').length - 2,
      is_https: originalUrl.startsWith('https://'),
      entropy: data.entropy || null,
      suspicious_keywords: data.suspicious_keywords || []
    },
    hostInfo: data.host_info || data.host || {
      ip: data.ip || 'Pending Host Intelligence Query',
      asn: data.asn || null,
      country: data.country || null,
      registrar: data.registrar || null
    },
    redirects: data.redirects || data.redirect_chain || [],
    suspiciousIndicators: data.suspicious_indicators || data.indicators || [],
    explainabilitySummary:
      data.explainability_summary ||
      data.explanation ||
      (data.top_features
        ? `Model identified ${Object.keys(data.top_features).length} dominant indicators driving classification.`
        : null),
    gnnGraphMeta: data.graph_meta || null,
    rawBackendData: data
  };
}

export default threatService;
