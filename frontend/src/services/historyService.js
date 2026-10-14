import apiClient from './api';

const LOCAL_STORAGE_KEY = 'cyberx_scan_history_v1';

export const historyService = {
  /**
   * Fetch scan history from backend if available, fallback to local storage
   */
  async getScanHistory() {
    try {
      const response = await apiClient.get('/api/history', { timeout: 3000 });
      if (response.data && Array.isArray(response.data)) {
        return {
          source: 'backend',
          scans: response.data
        };
      }
    } catch {
      // Fallback gracefully to client session history
    }

    // Load from local storage
    try {
      const local = localStorage.getItem(LOCAL_STORAGE_KEY);
      const parsed = local ? JSON.parse(local) : [];
      return {
        source: 'local',
        scans: parsed
      };
    } catch {
      return { source: 'local', scans: [] };
    }
  },

  /**
   * Save a completed scan to history
   */
  async saveScan(scanData) {
    if (!scanData) return;

    // Try posting to backend
    try {
      await apiClient.post('/api/history', scanData, { timeout: 2000 }).catch(() => {});
    } catch {
      // Continue to local save
    }

    // Persist locally
    try {
      const existing = this.getLocalScans();
      // Prepend and limit to 100 scans
      const updated = [scanData, ...existing.filter(s => s.id !== scanData.id)].slice(0, 100);
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
    } catch (e) {
      console.warn('Could not save to localStorage', e);
    }
  },

  getLocalScans() {
    try {
      const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  },

  clearHistory() {
    try {
      localStorage.removeItem(LOCAL_STORAGE_KEY);
    } catch (e) {
      console.warn(e);
    }
  }
};

export default historyService;
