import { apiClient } from './api';

/**
 * Service to manage Hugging Face Hub model synchronization,
 * checkpoint inspection, and memory cache warming.
 */

export const getHfModelStatus = async () => {
  const response = await apiClient.get('/api/models/hf-status');
  return response.data;
};

export const loadAllModelsFromHf = async (forceDownload = false) => {
  const response = await apiClient.post('/api/models/load-from-hf', {
    force_download: forceDownload,
  });
  return response.data;
};

export const loadSpecificModelFromHf = async (filename, forceDownload = false) => {
  const response = await apiClient.post('/api/models/load-from-hf', {
    models: [filename],
    force_download: forceDownload,
  });
  return response.data;
};

export const reloadModelMemoryCache = async () => {
  const response = await apiClient.post('/api/models/reload-cache');
  return response.data;
};

export default {
  getHfModelStatus,
  loadAllModelsFromHf,
  loadSpecificModelFromHf,
  reloadModelMemoryCache,
};
