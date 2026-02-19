/**
 * AuthContext — manages authentication state for web mode.
 *
 * In Electron mode (isElectron=true), auth is bypassed — the app works locally.
 * In web mode, users must authenticate via JWT before accessing features.
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { isElectron, getAccessToken, setServerUrl, getServerUrl } from '../api/client';
import { login as apiLogin, register as apiRegister, getMe, logout as apiLogout, checkServerHealth } from '../api/auth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // In Electron mode, skip auth entirely
  const skipAuth = isElectron;

  // Try to restore session from stored token
  useEffect(() => {
    if (skipAuth) {
      setUser({ id: 0, username: 'local', role: 'admin' });
      setLoading(false);
      return;
    }

    const token = getAccessToken();
    if (!token || !getServerUrl()) {
      setLoading(false);
      return;
    }

    getMe()
      .then((data) => {
        setUser(data.user || data);
      })
      .catch(() => {
        // Token expired or invalid — will need to re-login
        setUser(null);
      })
      .finally(() => setLoading(false));
  }, [skipAuth]);

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
