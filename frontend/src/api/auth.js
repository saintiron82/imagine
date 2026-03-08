/**
 * Auth API — register, login, refresh, me.
 */

import { apiClient, setTokens, clearTokens, getServerUrl } from './client';

export async function register({ server_password, username, email, password }) {
  const data = await apiClient.post('/api/v1/auth/register', {
    server_password,
    username,
    email,
    password,
  });
  if (data.access_token) {
    setTokens(data.access_token, data.refresh_token);
  }
  return data;
}

export async function login({ server_password, username, email, password }) {
  const data = await apiClient.post('/api/v1/auth/login', { server_password, username, email, password });
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
export function storeWorkerCredentials(server_password, username, password) {
  _workerCredentials = { server_password, username, password };
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

/**
 * Get public server info (no auth required).
 * Returns { group_name, initialized, version }.
 */
export async function getServerInfo(baseUrl) {
  try {
    const base = baseUrl || getServerUrl();
    if (!base) return { ok: false, error: 'No server URL' };

    const resp = await fetch(`${base}/api/v1/server/info`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    if (resp.ok) {
      const data = await resp.json();
      return { ok: true, ...data };
    }
    return { ok: false, error: `HTTP ${resp.status}` };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

/**
 * Initialize server (first-time setup).
 */
export async function initServer(baseUrl, { group_name, server_password, admin_username, admin_password }) {
  const resp = await fetch(`${baseUrl}/api/v1/server/init`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group_name, server_password, admin_username, admin_password }),
    signal: AbortSignal.timeout(10000),
  });
  let data;
  try {
    data = await resp.json();
  } catch {
    throw new Error(`Server error (HTTP ${resp.status})`);
  }
  if (!resp.ok) {
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  // Don't store tokens — user must login manually after group creation
  return data;
}

// ── Firebase Auth integration ───────────────────────────────

/**
 * Login to group server using Firebase ID Token.
 * Server verifies membership and returns server session JWT.
 */
export async function firebaseLogin(idToken) {
  const data = await apiClient.post('/api/v1/auth/firebase-login', {
    id_token: idToken,
  });
  if (data.access_token) {
    setTokens(data.access_token, data.refresh_token);
  }
  return data;
}

/**
 * Join a group using invite code + Firebase ID Token.
 */
export async function joinGroup(idToken, inviteCode) {
  const data = await apiClient.post('/api/v1/auth/join', {
    id_token: idToken,
    invite_code: inviteCode,
  });
  if (data.access_token) {
    setTokens(data.access_token, data.refresh_token);
  }
  return data;
}

/**
 * Initialize server with Firebase Auth (Electron create group).
 */
export async function firebaseInitServer(baseUrl, { group_name, id_token }) {
  const resp = await fetch(`${baseUrl}/api/v1/server/firebase-init`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ group_name, id_token }),
    signal: AbortSignal.timeout(10000),
  });
  let data;
  try {
    data = await resp.json();
  } catch {
    throw new Error(`Server error (HTTP ${resp.status})`);
  }
  if (!resp.ok) {
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  return data;
}

/**
 * Reset group (auth data). File data is preserved.
 * Requires server password for security.
 */
export async function resetGroup(baseUrl, serverPassword) {
  const resp = await fetch(`${baseUrl}/api/v1/server/reset-group`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ server_password: serverPassword }),
    signal: AbortSignal.timeout(10000),
  });
  let data;
  try {
    data = await resp.json();
  } catch {
    throw new Error(`Server error (HTTP ${resp.status})`);
  }
  if (!resp.ok) {
    throw new Error(data.detail || `HTTP ${resp.status}`);
  }
  return data;
}
