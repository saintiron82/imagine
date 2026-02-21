/**
 * API Client — fetch wrapper with JWT auto-attach and token refresh.
 *
 * Usage:
 *   import { apiClient } from '../api/client';
 *   const data = await apiClient.get('/search/triaxis', { query: '...' });
 */

const TOKEN_KEY = 'imagine-access-token';
const REFRESH_KEY = 'imagine-refresh-token';
const SERVER_URL_KEY = 'imagine-server-url';

/** Detect if running inside Electron */
export const isElectron = typeof window !== 'undefined' && !!window.electron;

/** Get server base URL */
export function getServerUrl() {
  if (typeof window !== 'undefined') {
    return localStorage.getItem(SERVER_URL_KEY) || '';
  }
  return '';
}

export function setServerUrl(url) {
  localStorage.setItem(SERVER_URL_KEY, url.replace(/\/+$/, ''));
}

/** Token management */
export function getAccessToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function getRefreshToken() {
  return localStorage.getItem(REFRESH_KEY);
}

export function setTokens(access, refresh) {
  if (access) localStorage.setItem(TOKEN_KEY, access);
  if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
}

export function clearTokens() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(REFRESH_KEY);
}

/** Forward refreshed tokens to the embedded worker process (Electron only). */
function _syncTokensToWorker(accessToken, refreshToken) {
  if (typeof window !== 'undefined' && window.electron?.worker?.updateTokens) {
    window.electron.worker.updateTokens({ accessToken, refreshToken })
      .catch(() => { /* ignore — worker may not be running */ });
  }
}

/** Pending refresh promise to avoid concurrent refresh calls */
let _refreshPromise = null;

async function refreshAccessToken() {
  if (_refreshPromise) return _refreshPromise;

  const refresh = getRefreshToken();
  if (!refresh) {
    clearTokens();
    return null;
  }

  _refreshPromise = (async () => {
    try {
      const base = getServerUrl();
      const resp = await fetch(`${base}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh_token: refresh }),
      });

      if (resp.ok) {
        const data = await resp.json();
        setTokens(data.access_token, data.refresh_token);
        // Forward new tokens to embedded worker (if running in Electron)
        _syncTokensToWorker(data.access_token, data.refresh_token);
        return data.access_token;
      } else {
        clearTokens();
        return null;
      }
    } catch {
      clearTokens();
      return null;
    } finally {
      _refreshPromise = null;
    }
  })();

  return _refreshPromise;
}

/**
 * Core fetch wrapper with auth and retry.
 */
async function request(method, path, { body, params, raw } = {}) {
  const base = getServerUrl();
  let url = `${base}${path}`;

  if (params) {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) qs.set(k, v);
    }
    const qstr = qs.toString();
    if (qstr) url += `?${qstr}`;
  }

  const headers = {};
  const token = getAccessToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;

  let fetchOpts = { method, headers };

  if (body !== undefined) {
    if (body instanceof FormData) {
      fetchOpts.body = body;
      // Let browser set Content-Type with boundary
    } else {
      headers['Content-Type'] = 'application/json';
      fetchOpts.body = JSON.stringify(body);
    }
  }

  let resp = await fetch(url, fetchOpts);

  // Auto-refresh on 401
  if (resp.status === 401) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      headers['Authorization'] = `Bearer ${newToken}`;
      fetchOpts.headers = headers;
      resp = await fetch(url, fetchOpts);
    }
  }

  if (raw) return resp;

  if (!resp.ok) {
    let detail = '';
    try {
      const err = await resp.json();
      detail = err.detail || JSON.stringify(err);
    } catch {
      detail = resp.statusText;
    }
    throw new ApiError(resp.status, detail);
  }

  return resp.json();
}

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

/** Public API client */
export const apiClient = {
  get: (path, params) => request('GET', path, { params }),
  post: (path, body) => request('POST', path, { body }),
  patch: (path, body) => request('PATCH', path, { body }),
  delete: (path) => request('DELETE', path),
  upload: (path, formData) => request('POST', path, { body: formData }),
  raw: (method, path, opts) => request(method, path, { ...opts, raw: true }),
};
