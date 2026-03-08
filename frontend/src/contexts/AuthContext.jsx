/**
 * AuthContext — manages authentication state.
 *
 * Two-layer auth:
 *   1. Firebase Auth (personal account, persistent session)
 *   2. Group server JWT (membership-based, issued after Firebase login)
 *
 * Electron local mode: localhost auto-admin (no Firebase needed)
 */

import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { isElectron, getAccessToken, setServerUrl, getServerUrl, setTokens, clearTokens } from '../api/client';
import {
  getMe, logout as apiLogout, checkServerHealth,
  storeWorkerCredentials, clearWorkerCredentials,
  firebaseLogin, joinGroup, getServerInfo,
} from '../api/auth';
import { onAuthStateChanged, getIdToken, signOut as firebaseSignOut } from '../api/firebaseAuth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  // Firebase user (personal account)
  const [firebaseUser, setFirebaseUser] = useState(null);
  const [firebaseLoading, setFirebaseLoading] = useState(true);

  // Group server user (after membership verified)
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // 'server' | 'client' | null
  const [authMode, setAuthMode] = useState(null);

  // Connected group info
  const [connectedGroup, setConnectedGroup] = useState(null);

  // Track if we should skip Firebase (Electron localhost auto-admin)
  const skipFirebase = useRef(false);

  // ── Firebase Auth state listener ──────────────────────────
  useEffect(() => {
    const unsub = onAuthStateChanged((fbUser) => {
      setFirebaseUser(fbUser);
      setFirebaseLoading(false);

      if (!fbUser) {
        // Firebase signed out — clear server session too
        clearTokens();
        clearWorkerCredentials();
        setUser(null);
        setLoading(false);
        return;
      }

      // Auto-reconnect to last group
      const lastGroup = localStorage.getItem('imagine-last-group');
      const lastGroupUrl = localStorage.getItem('imagine-last-group-url');
      if (lastGroup && lastGroupUrl && !skipFirebase.current) {
        connectToGroup(lastGroupUrl, lastGroup).finally(() => setLoading(false));
      } else {
        setLoading(false);
      }
    });

    return unsub;
  }, []);

  // ── Connect to group server with Firebase token ───────────
  const connectToGroup = useCallback(async (serverUrl, groupName) => {
    setError('');
    try {
      setServerUrl(serverUrl);
      const idToken = await getIdToken();
      if (!idToken) throw new Error('Firebase not authenticated');

      const data = await firebaseLogin(idToken);
      if (data.access_token) {
        const me = await getMe();
        setUser(me.user || me);
        setConnectedGroup(groupName);
        localStorage.setItem('imagine-last-group', groupName);
        localStorage.setItem('imagine-last-group-url', serverUrl);
        return true;
      }
      return false;
    } catch (e) {
      const detail = e.detail || e.message || 'Connection failed';
      // Don't set error for auto-reconnect failures
      if (detail !== 'Not a member of this group') {
        setError(detail);
      }
      setUser(null);
      return false;
    }
  }, []);

  // ── Join group with invite code ───────────────────────────
  const joinGroupWithCode = useCallback(async (serverUrl, inviteCode, groupName) => {
    setError('');
    try {
      setServerUrl(serverUrl);
      const idToken = await getIdToken();
      if (!idToken) throw new Error('Firebase not authenticated');

      const data = await joinGroup(idToken, inviteCode);
      if (data.access_token) {
        const me = await getMe();
        setUser(me.user || me);
        setConnectedGroup(groupName);
        localStorage.setItem('imagine-last-group', groupName);
        localStorage.setItem('imagine-last-group-url', serverUrl);
        return true;
      }
      return false;
    } catch (e) {
      setError(e.detail || e.message || 'Join failed');
      return false;
    }
  }, []);

  // ── Legacy login (for backward compatibility / Electron local) ──
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

  // ── Legacy register ───────────────────────────────────────
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

  // ── Logout (both Firebase and server) ─────────────────────
  const logout = useCallback(async () => {
    apiLogout();
    clearWorkerCredentials();
    setUser(null);
    setConnectedGroup(null);
    localStorage.removeItem('imagine-last-group');
    localStorage.removeItem('imagine-last-group-url');
    skipFirebase.current = false;
  }, []);

  // ── Disconnect from group (keep Firebase session) ─────────
  const disconnectGroup = useCallback(() => {
    clearTokens();
    clearWorkerCredentials();
    setUser(null);
    setConnectedGroup(null);
    localStorage.removeItem('imagine-last-group');
    localStorage.removeItem('imagine-last-group-url');
  }, []);

  // ── Full logout (Firebase + server) ───────────────────────
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
    // Firebase layer
    firebaseUser,
    firebaseLoading,
    isFirebaseAuthenticated: !!firebaseUser,

    // Group server layer
    user,
    loading: loading || firebaseLoading,
    error,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
    connectedGroup,

    // Auth mode
    authMode,
    skipAuth: false,

    // Actions
    login,              // Legacy (server_password based)
    register,           // Legacy
    logout,             // Server session only
    fullLogout,         // Firebase + server
    disconnectGroup,    // Leave group, keep Firebase
    connectToGroup,     // Firebase token → server JWT
    joinGroupWithCode,  // Invite code + Firebase token
    checkServerHealth,
    configureAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
