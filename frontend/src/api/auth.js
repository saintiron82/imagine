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

/**
 * Test server connection (unauthenticated health check).
 */
export async function checkServerHealth() {
  try {
    const base = getServerUrl();
    if (!base) return { ok: false, error: 'No server URL configured' };

    const resp = await fetch(`${base}/api/v1/health`, { method: 'GET' });
    if (resp.ok) {
      const data = await resp.json();
      return { ok: true, version: data.version };
    }
    return { ok: false, error: `HTTP ${resp.status}` };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}
