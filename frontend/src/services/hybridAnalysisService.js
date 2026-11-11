import { apiClient } from './api';

export const hybridAnalysisService = {
  /**
   * Check connection and quota status for Hybrid Analysis (Falcon Sandbox)
   */
  async getStatus() {
    const response = await apiClient.get('/api/hybrid-analysis/status');
    return response.data;
  },

  /**
   * Fetch recent threat detonations from the global live feed
   */
  async getFeed(limit = 25) {
    const response = await apiClient.get(`/api/hybrid-analysis/feed?limit=${limit}`);
    return response.data;
  },

  /**
   * Query multi-engine AV scan summary and threat verdict by SHA256 hash
   */
  async lookupHash(sha256) {
    const clean = sha256.trim();
    const response = await apiClient.get(`/api/hybrid-analysis/hash/${encodeURIComponent(clean)}`);
    return response.data;
  },

  /**
   * Retrieve list of supported sandbox virtual environments
   */
  async getEnvironments() {
    const response = await apiClient.get('/api/hybrid-analysis/environments');
    return response.data;
  },
};

export default hybridAnalysisService;
