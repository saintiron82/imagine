/**
 * AuthContext — manages 2-layer authentication state.
 *
 * Layer 1: Firebase Auth (personal identity — Google / email+password)
 * Layer 2: Server JWT (group access — server_name + server_password via /auth/connect)
 *
 * Electron local mode: localhost auto-admin (no Firebase needed)
 */

import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { isElectron, getAccessToken, setServerUrl, getServerUrl, setTokens, clearTokens } from '../api/client';
import {
  getMe, logout as apiLogout, checkServerHealth,
  storeWorkerCredentials, clearWorkerCredentials,
  firebaseConnect, getServerInfo,
} from '../api/auth';
import { onAuthStateChanged, getIdToken, signOut as firebaseSignOut } from '../api/firebaseAuth';
import { lookupGroup } from '../api/firebase';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  // Layer 1: Firebase user (personal identity)
  const [firebaseUser, setFirebaseUser] = useState(null);
  const [firebaseLoading, setFirebaseLoading] = useState(true);

  // Layer 2: Server user (group access, after /auth/connect)
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // 'server' | 'client' | null
  const [authMode, setAuthMode] = useState(null);

  // Connected server info { name, url }
  const [connectedServer, setConnectedServer] = useState(null);

  // Track if we should skip Firebase (Electron localhost auto-admin)
  const skipFirebase = useRef(false);

  // ── On mount: clear server tokens (require fresh login every session) ──
  useEffect(() => {
    clearTokens();
    clearWorkerCredentials();
  }, []);

  // ── Layer 1: Firebase Auth state listener ──────────────────
  useEffect(() => {
    const unsub = onAuthStateChanged((fbUser) => {
      setFirebaseUser(fbUser);
      setFirebaseLoading(false);

      if (!fbUser) {
        clearTokens();
        clearWorkerCredentials();
        setUser(null);
        setConnectedServer(null);
        setLoading(false);
        return;
      }

      // Firebase signed in but server connection required separately
      setUser(null);
      setLoading(false);
    });

    return unsub;
  }, []);

  // ── Internal: reconnect to a known server URL ──────────────
  const _reconnectServer = useCallback(async (serverUrl, serverName, serverPassword) => {
    try {
      setServerUrl(serverUrl);
      const idToken = await getIdToken();
      if (!idToken) return false;

      await firebaseConnect(idToken, serverPassword);
      const me = await getMe();
      setUser(me.user || me);
      setConnectedServer({ name: serverName, url: serverUrl });
      return true;
    } catch {
      // Silent fail on auto-reconnect
      setUser(null);
      setConnectedServer(null);
      return false;
    }
  }, []);

  // ── Layer 2: Connect to server (RTDB lookup + /auth/connect) ──
  const connectToServer = useCallback(async (serverName, serverPassword, directUrl) => {
    setError('');
    try {
      // Build candidate URL list
      let urls;
      if (directUrl) {
        urls = [directUrl];
      } else {
        const info = await lookupGroup(serverName);
        if (!info) throw new Error('Server not found');
        urls = info.urls || [info.url];
      }

      // Get Firebase ID token for Layer 1 identity
      const idToken = await getIdToken();
      if (!idToken) throw new Error('Firebase not authenticated');

      // Try each URL until one responds
      let lastErr = null;
      for (const candidateUrl of urls) {
        try {
          // Quick health check (3s timeout)
          const hResp = await fetch(`${candidateUrl}/api/v1/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(3000),
          });
          if (!hResp.ok) continue;

          // Health OK → authenticate
          setServerUrl(candidateUrl);
          await firebaseConnect(idToken, serverPassword);
          const me = await getMe();
          setUser(me.user || me);
          setConnectedServer({ name: serverName, url: candidateUrl });

          // Remember for auto-reconnect
          localStorage.setItem('imagine-last-server', serverName);
          localStorage.setItem('imagine-last-server-url', candidateUrl);
          sessionStorage.setItem('imagine-last-server-pw', serverPassword);

          return true;
        } catch (e) {
          lastErr = e;
          // Try next URL
        }
      }

      // All URLs failed
      throw lastErr || new Error('Server unreachable');
    } catch (e) {
      const detail = e.detail || e.message || 'Connection failed';
      setError(detail);
      setUser(null);
      setConnectedServer(null);
      return false;
    }
  }, []);

  // ── Legacy login (Electron local mode / backward compat) ───
  const login = useCallback(async ({ server_password, username, password, serverUrl }) => {
    setError('');
    try {
      if (serverUrl) setServerUrl(serverUrl);
      const { login: apiLogin } = await import('../api/auth');
      const data = await apiLogin({ server_password, username, password });
      storeWorkerCredentials(server_password, username, password);
      const me = await getMe();
      setUser(me.user || me);
      skipFirebase.current = true;
      return true;
    } catch (e) {
      setError(e.detail || e.message || 'Login failed');
      return false;
    }
  }, []);

  // ── Legacy register ────────────────────────────────────────
  const register = useCallback(async ({ server_password, username, email, password, serverUrl }) => {
    setError('');
    try {
      if (serverUrl) setServerUrl(serverUrl);
      const { register: apiRegister } = await import('../api/auth');
      await apiRegister({ server_password, username, email, password });
      const me = await getMe();
      setUser(me.user || me);
      return true;
    } catch (e) {
      setError(e.detail || e.message || 'Registration failed');
      return false;
    }
  }, []);

  // ── Disconnect from server (keep Firebase session) ─────────
  const disconnectServer = useCallback(() => {
    clearTokens();
    clearWorkerCredentials();
    setUser(null);
    setConnectedServer(null);
    localStorage.removeItem('imagine-last-server');
    localStorage.removeItem('imagine-last-server-url');
    sessionStorage.removeItem('imagine-last-server-pw');
  }, []);

  // ── Logout (server session only) ───────────────────────────
  const logout = useCallback(async () => {
    apiLogout();
    clearWorkerCredentials();
    disconnectServer();
    skipFirebase.current = false;
  }, [disconnectServer]);

  // ── Full logout (Firebase + server) ────────────────────────
  const fullLogout = useCallback(async () => {
    await logout();
    try {
      await firebaseSignOut();
    } catch {
      // Ignore Firebase sign out errors
    }
  }, [logout]);

  const configureAuth = useCallback(async (mode) => {
    setAuthMode(mode);
    const token = getAccessToken();
    if (token) {
      try {
        const me = await getMe();
        setUser(me.user || me);
        return;
      } catch {
        // Token invalid
      }
    }
    if (mode === 'client') clearWorkerCredentials();
    setUser(null);
  }, []);

  const value = {
    // Layer 1: Firebase identity
    firebaseUser,
    firebaseLoading,
    isFirebaseAuthenticated: !!firebaseUser,

    // Layer 2: Server access
    user,
    loading: loading || firebaseLoading,
    error,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
    connectedServer,

    // Auth mode
    authMode,
    skipAuth: false,

    // Actions
    connectToServer,    // Layer 2: RTDB lookup + /auth/connect
    disconnectServer,   // Leave server, keep Firebase
    login,              // Legacy (server_password based)
    register,           // Legacy
    logout,             // Server session only
    fullLogout,         // Firebase + server
    checkServerHealth,
    configureAuth,
    getServerInfo,
    clearError: () => setError(''),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
