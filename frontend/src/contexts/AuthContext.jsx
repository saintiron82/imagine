/**
 * AuthContext — manages authentication state.
 *
 * - Electron server mode: auth bypassed (local admin)
 * - Electron client mode: JWT auth required (remote server)
 * - Web mode: JWT auth required
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { isElectron, getAccessToken, setServerUrl, getServerUrl } from '../api/client';
import { login as apiLogin, register as apiRegister, getMe, logout as apiLogout, checkServerHealth } from '../api/auth';

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

    // Electron: determine auth mode from config.yaml
    const determineAuth = async () => {
      try {
        const result = await window.electron?.pipeline?.getConfig();
        const mode = result?.config?.app?.mode;

        if (mode === 'client') {
          // Client mode: JWT auth required
          setSkipAuth(false);
          if (result?.config?.app?.server_url) {
            setServerUrl(result.config.app.server_url);
          }
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
          // Server mode or unconfigured: local bypass
          setSkipAuth(true);
          setUser({ id: 0, username: 'local', role: 'admin' });
        }
      } catch {
        // Fallback to local bypass on error
        setSkipAuth(true);
        setUser({ id: 0, username: 'local', role: 'admin' });
      }
      setLoading(false);
    };

    determineAuth();
  }, []);

  const login = useCallback(async ({ username, password, serverUrl }) => {
    setError('');
    try {
      if (serverUrl) setServerUrl(serverUrl);
      const data = await apiLogin({ username, password });
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
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
