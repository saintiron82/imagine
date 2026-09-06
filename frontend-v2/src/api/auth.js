/**
 * Auth API — Firebase connect, me, server init.
 */

import { apiClient, setTokens } from './client';

function buildInitHeaders(setupToken) {
  const headers = { 'Content-Type': 'application/json' };
  const token = (setupToken || '').trim();
  if (token) {
    headers['X-Imagine-Setup-Token'] = token;
  }
  return headers;
}

export async function getMe() {
  return apiClient.get('/api/v1/auth/me');
}

/**
 * Initialize server (first-time setup).
 */
export async function initServer(baseUrl, {
  group_name,
  server_password,
  admin_username,
  admin_password,
  firebase_uid,
  firebase_email,
  setupToken,
  setup_token,
}) {
  const body = { group_name, server_password, admin_username, admin_password };
  if (firebase_uid) body.firebase_uid = firebase_uid;
  if (firebase_email) body.firebase_email = firebase_email;
  const resp = await fetch(`${baseUrl}/api/v1/server/init`, {
    method: 'POST',
    headers: buildInitHeaders(setupToken || setup_token),
    body: JSON.stringify(body),
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
  // Return first-run tokens; the caller stores them only after post-init work succeeds.
  return data;
}

// ── Firebase Auth integration ───────────────────────────────

/**
 * Connect to server using Firebase ID Token + server password (2-layer auth).
 * Server verifies both credentials and returns session JWT.
 */
export async function firebaseConnect(idToken, serverPassword) {
  const data = await apiClient.post('/api/v1/auth/connect', {
    firebase_id_token: idToken,
    server_password: serverPassword,
  });
  if (data.access_token) {
    setTokens(data.access_token, data.refresh_token);
  }
  return data;
}
