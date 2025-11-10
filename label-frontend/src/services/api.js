import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // You can add auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response.data;
  },
  (error) => {
    const message = error.response?.data?.message || error.message || 'An error occurred';
    console.error('API Error:', message);
    return Promise.reject(error);
  }
);

// Label API endpoints
export const labelAPI = {
  // Get all labels with optional filters
  getLabels: (params = {}) => {
    return api.get('/labels', { params });
  },

  // Get label tree
  getTree: () => {
    return api.get('/labels/tree');
  },

  // Get label by ID
  getLabel: (id) => {
    return api.get(`/labels/${id}`);
  },

  // Get children of a label
  getChildren: (id) => {
    return api.get(`/labels/${id}/children`);
  },

  // Create new label
  createLabel: (data) => {
    return api.post('/labels', data);
  },

  // Update label
  updateLabel: (id, data) => {
    return api.put(`/labels/${id}`, data);
  },

  // Delete label
  deleteLabel: (id) => {
    return api.delete(`/labels/${id}`);
  },

  // Health check
  healthCheck: () => {
    return api.get('/health');
  },
};

export default api;


