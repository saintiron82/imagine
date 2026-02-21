/**
 * Auth API — register, login, refresh, me.
 */

import { apiClient, setTokens, clearTokens, getServerUrl } from './client';

export async function register({ invite_code, username, email, password }) {
  const data = await apiClient.post('/api/v1/auth/register', {
    invite_code,
    username,
    email,
    password,
  });
  if (data.access_token) {
    setTokens(data.access_token, data.refresh_token);
  }
  return data;
}

export async function login({ username, email, password }) {
  const data = await apiClient.post('/api/v1/auth/login', { username, email, password });
  if (data.access_token) {
    setTokens(data.access_token, data.refresh_token);
  }
  return data;
}

export async function getMe() {
  return apiClient.get('/api/v1/auth/me');
}

export function logout() {
  clearTokens();
}

/** In-memory credential store for worker auth (not persisted to disk). */
let _workerCredentials = null;

/**
 * Store login credentials in memory (for worker to do independent login).
 * Called after successful login. Never persisted to localStorage.
 */
export function storeWorkerCredentials(username, password) {
  _workerCredentials = { username, password };
}

/** Retrieve stored worker credentials (one-time read, still kept in memory). */
export function getWorkerCredentials() {
  return _workerCredentials;
}

/** Clear stored worker credentials (on logout). */
export function clearWorkerCredentials() {
  _workerCredentials = null;
}

/**
 * Test server connection (unauthenticated health check).
 */
export async function checkServerHealth() {
  try {
    const base = getServerUrl();
    if (!base) return { ok: false, error: 'No server URL configured' };

    const resp = await fetch(`${base}/api/v1/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    if (resp.ok) {
      const data = await resp.json();
      return { ok: true, version: data.version, serverName: data.server_name || null };
    }
    return { ok: false, error: `HTTP ${resp.status}` };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}
