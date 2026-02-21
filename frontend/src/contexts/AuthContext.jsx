/**
 * AuthContext — manages authentication state.
 *
 * - Electron server mode: auth bypassed (local admin)
 * - Electron client mode: JWT auth required (remote server)
 * - Web mode: JWT auth required
 *
 * Mode is determined at runtime by SetupPage selection (not config.yaml).
 * App.jsx calls configureAuth(mode) after user picks a mode.
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { isElectron, getAccessToken, setServerUrl, getServerUrl } from '../api/client';
import { login as apiLogin, register as apiRegister, getMe, logout as apiLogout, checkServerHealth, storeWorkerCredentials, clearWorkerCredentials } from '../api/auth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // null = undetermined, true = bypass, false = JWT required
  const [skipAuth, setSkipAuth] = useState(null);

  useEffect(() => {
    // Web mode: always require JWT
    if (!isElectron) {
      setSkipAuth(false);
      const token = getAccessToken();
      if (!token || !getServerUrl()) {
        setLoading(false);
        return;
      }
      getMe()
        .then((data) => setUser(data.user || data))
        .catch(() => setUser(null))
        .finally(() => setLoading(false));
      return;
    }

    // Electron: start with local bypass (SetupPage will determine actual mode)
    setSkipAuth(true);
    setUser({ id: 0, username: 'local', role: 'admin' });
    setLoading(false);
  }, []);

  /**
   * Switch auth mode after SetupPage selection.
   * Called by App.jsx when user picks server or client mode.
   */
  const configureAuth = useCallback(async (mode) => {
    if (mode === 'client') {
      setSkipAuth(false);
      setUser(null);
      // Try to restore existing session
      const token = getAccessToken();
      if (token && getServerUrl()) {
        try {
          const data = await getMe();
          setUser(data.user || data);
        } catch {
          setUser(null);
        }
      }
    } else {
      // Server mode or reset: local admin bypass
      setSkipAuth(true);
      setUser({ id: 0, username: 'local', role: 'admin' });
    }
  }, []);

  const login = useCallback(async ({ username, password, serverUrl }) => {
    setError('');
    try {
      if (serverUrl) setServerUrl(serverUrl);
      const data = await apiLogin({ username, password });
      // Store credentials in memory so worker can do independent login
      storeWorkerCredentials(username, password);
      const me = await getMe();
      setUser(me.user || me);
      return true;
    } catch (e) {
      setError(e.detail || e.message || 'Login failed');
      return false;
    }
  }, []);

  const register = useCallback(async ({ invite_code, username, email, password, serverUrl }) => {
    setError('');
    try {
      if (serverUrl) setServerUrl(serverUrl);
      await apiRegister({ invite_code, username, email, password });
      const me = await getMe();
      setUser(me.user || me);
      return true;
    } catch (e) {
      setError(e.detail || e.message || 'Registration failed');
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    apiLogout();
    clearWorkerCredentials();
    setUser(null);
  }, []);

  const value = {
    user,
    loading,
    error,
    isAuthenticated: !!user,
    isAdmin: user?.role === 'admin',
    skipAuth,
    login,
    register,
    logout,
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
