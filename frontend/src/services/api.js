import axios from 'axios';

// Get base URL from environment variable or fallback to standard FastAPI local address
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 25000, // 25s timeout for AI inference & graph analysis
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Inject custom headers or tracking IDs if necessary
    config.headers['X-Client-Platform'] = 'CyberX-ExplainableAI-Frontend/1.0';
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    let errorMessage = 'An unexpected network error occurred.';
    let isOffline = false;

    if (error.code === 'ECONNABORTED') {
      errorMessage = 'Request timed out. The backend model analysis took too long to respond.';
    } else if (!error.response) {
      errorMessage = `Cannot connect to FastAPI backend at ${BASE_URL}. Ensure Uvicorn is running.`;
      isOffline = true;
    } else if (error.response.status === 404) {
      errorMessage = `Endpoint ${error.config?.url} is not yet implemented or route was not found on backend.`;
    } else if (error.response.status === 422) {
      const detail = error.response.data?.detail;
      errorMessage = Array.isArray(detail)
        ? detail.map((d) => `${d.loc ? d.loc.join('.') + ': ' : ''}${d.msg}`).join('; ')
        : 'Invalid request data validation error.';
    } else if (error.response.status >= 500) {
      errorMessage = error.response.data?.detail || error.response.data?.message || 'Internal Server Error in ML pipeline.';
    } else if (error.response.data?.detail) {
      errorMessage = error.response.data.detail;
    }

    const enhancedError = new Error(errorMessage);
    enhancedError.status = error.response?.status;
    enhancedError.isOffline = isOffline;
    enhancedError.originalError = error;
    enhancedError.data = error.response?.data;

    return Promise.reject(enhancedError);
  }
);

// Method to verify backend health
export const checkBackendHealth = async () => {
  const startTime = performance.now();
  try {
    // Attempt standard FastAPI health endpoints in order: /health, /api/health, /
    const endpoints = ['/health', '/api/health', '/'];
    for (const ep of endpoints) {
      try {
        const res = await apiClient.get(ep, { timeout: 3500 });
        const latency = Math.round(performance.now() - startTime);
        return {
          online: true,
          status: res.status,
          data: res.data,
          latencyMs: latency,
          endpointUsed: ep,
          url: apiClient.defaults.baseURL,
        };
      } catch (err) {
        if (err.status && err.status < 500) {
          // If server responded (even 404 or 405), backend is reachable!
          const latency = Math.round(performance.now() - startTime);
          return {
            online: true,
            status: err.status,
            latencyMs: latency,
            endpointUsed: ep,
            url: apiClient.defaults.baseURL,
          };
        }
      }
    }
    return {
      online: false,
      message: `No response from ${apiClient.defaults.baseURL}`,
      latencyMs: null,
      url: apiClient.defaults.baseURL,
    };
  } catch {
    return {
      online: false,
      message: `Failed to connect to ${apiClient.defaults.baseURL}`,
      latencyMs: null,
      url: apiClient.defaults.baseURL,
    };
  }
};

export const setApiBaseUrl = (newUrl) => {
  if (newUrl) {
    apiClient.defaults.baseURL = newUrl.replace(/\/+$/, '');
  }
};

export const getApiBaseUrl = () => {
  return apiClient.defaults.baseURL;
};

export default apiClient;
