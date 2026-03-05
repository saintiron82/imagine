/**
 * AuthContext — manages authentication state.
 *
 * All modes (Electron + Web) require JWT authentication.
 * LoginPage is always shown first; role-based mode is set after login.
 */

import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { isElectron, getAccessToken, setServerUrl, getServerUrl } from '../api/client';
import { login as apiLogin, register as apiRegister, getMe, logout as apiLogout, checkServerHealth, storeWorkerCredentials, clearWorkerCredentials, getServerInfo } from '../api/auth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // null = undetermined, true = bypass, false = JWT required
  const [skipAuth, setSkipAuth] = useState(null);

  // 'server' | 'client' | null — lets LoginPage adapt its UI
  const [authMode, setAuthMode] = useState(null);

  useEffect(() => {
    // All modes: require JWT. Try auto-login with existing token.
    setSkipAuth(false);
    const token = getAccessToken();
    if (!token) {
      setLoading(false);
      return;
    }
    getMe()
      .then((data) => setUser(data.user || data))
      .catch(() => setUser(null))
      .finally(() => setLoading(false));
  }, []);

  /**
   * Re-check auth state. Called after login/init to refresh user profile.
   * If token exists, tries to load user; otherwise shows LoginPage.
   */
  const configureAuth = useCallback(async (mode) => {
    setAuthMode(mode);
    setSkipAuth(false);

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

  const register = useCallback(async ({ server_password, username, email, password, serverUrl }) => {
    setError('');
    try {
      if (serverUrl) setServerUrl(serverUrl);
      await apiRegister({ server_password, username, email, password });
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
    authMode,
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
