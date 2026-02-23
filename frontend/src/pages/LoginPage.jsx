/**
 * LoginPage — server discovery + login / register.
 *
 * Features:
 * - mDNS auto-discovery of Imagine servers (Electron only)
 * - Recent server history (all modes)
 * - Health check with server name display
 * - Login / Register with invite code
 * - Server mode: simplified UI (no URL selection, auto health check)
 */

import { useState, useEffect, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { setServerUrl as setClientServerUrl } from '../api/client';
import { useMdnsDiscovery } from '../hooks/useMdnsDiscovery';
import {
  getServerHistory,
  addServerToHistory,
  removeServerFromHistory,
  formatRelativeTime,
} from '../utils/serverHistory';
import {
  LogIn, UserPlus, Server, Eye, EyeOff, CheckCircle, XCircle,
  Download, Wifi, X, Clock, Radio, Loader,
} from 'lucide-react';

export default function LoginPage({ onShowDownload, serverRunning, serverPort }) {
  const { login, register, error, checkServerHealth, authMode } = useAuth();
  const { t, locale, setLocale, availableLocales } = useLocale();

  const isServerMode = authMode === 'server';

  const [mode, setMode] = useState('login');
  const [serverUrl, setServerUrlLocal] = useState(
    localStorage.getItem('imagine-server-url') || 'http://localhost:8000'
  );
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [inviteCode, setInviteCode] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [serverStatus, setServerStatus] = useState(null);
  const [serverError, setServerError] = useState('');
  const [serverName, setServerName] = useState('');

  // Server history
  const [history, setHistory] = useState([]);
  useEffect(() => {
    if (!isServerMode) setHistory(getServerHistory());
  }, [isServerMode]);

  // mDNS discovery (Electron only, not in server mode)
  const { servers: mdnsServers, browsing } = useMdnsDiscovery();

  // Auto-check on mount if URL is saved + auto-fill username (client/web mode only)
  useEffect(() => {
    if (isServerMode) return;
    const saved = localStorage.getItem('imagine-server-url');
    if (saved && saved !== 'http://localhost:8000') {
      handleCheckServer(saved);
      const entry = getServerHistory().find(h => h.url === saved);
      if (entry?.lastUsername) {
        setUsername(entry.lastUsername);
      }
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Server mode: auto-configure URL and health check when server is ready
  const healthCheckDone = useRef(false);
  useEffect(() => {
    if (!isServerMode) return;
    const url = `http://localhost:${serverPort || 8000}`;
    setServerUrlLocal(url);
    setClientServerUrl(url);

    // Auto-fill last username from server history
    const entry = getServerHistory().find(h => h.url === url);
    if (entry?.lastUsername && !username) {
      setUsername(entry.lastUsername);
    }

    if (serverRunning && !healthCheckDone.current) {
      healthCheckDone.current = true;
      // Small delay to ensure server is fully ready
      const timer = setTimeout(() => handleCheckServer(url), 500);
      return () => clearTimeout(timer);
    }
  }, [isServerMode, serverRunning, serverPort]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleCheckServer = async (url) => {
    const targetUrl = (url || serverUrl).trim();
    if (!targetUrl) return;

    setServerStatus('checking');
    setServerError('');
    setServerName('');
    setClientServerUrl(targetUrl);

    const result = await checkServerHealth();
    if (result.ok) {
      setServerStatus('ok');
      setServerName(result.serverName || '');
    } else {
      setServerStatus('error');
      setServerError(result.error);
    }
  };

  const handleSelectServer = (url) => {
    setServerUrlLocal(url);
    setServerStatus(null);
    handleCheckServer(url);
    // Auto-fill username from server history
    const entry = getServerHistory().find(h => h.url === url);
    if (entry?.lastUsername) {
      setUsername(entry.lastUsername);
    }
  };

  const handleSelectMdns = (server) => {
    const addr = server.addresses?.[0] || server.host;
    const url = `http://${addr}:${server.port}`;
    setServerUrlLocal(url);
    setServerStatus(null);
    handleCheckServer(url);
    // Auto-fill username from history if previously connected
    const entry = getServerHistory().find(h => h.url === url);
    if (entry?.lastUsername) {
      setUsername(entry.lastUsername);
    }
  };

  const handleRemoveHistory = (e, url) => {
    e.stopPropagation();
    removeServerFromHistory(url);
    setHistory(getServerHistory());
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);

    const trimmedUrl = serverUrl.trim();
    let success;

    if (mode === 'login') {
      success = await login({ username, password, serverUrl: trimmedUrl });
    } else {
      success = await register({
        invite_code: inviteCode,
        username,
        email: email || undefined,
        password,
        serverUrl: trimmedUrl,
      });
    }

    if (success) {
      addServerToHistory({
        url: trimmedUrl,
        name: serverName,
        version: '',
        lastUsername: username,
      });
    }

    setSubmitting(false);
  };

  // Filter history to exclude URLs already shown in mDNS
  const mdnsUrls = new Set(
    mdnsServers.map((s) => `http://${s.addresses?.[0]}:${s.port}`)
  );
  const filteredHistory = history.filter((h) => !mdnsUrls.has(h.url));

  // Server mode: waiting for server to start
  const serverModeWaiting = isServerMode && !serverRunning;
  // Server mode: server ready (health check passed or running)
  const serverModeReady = isServerMode && serverRunning;

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900">
      <div className="w-full max-w-md p-8">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-white mb-2">Imagine</h1>
          <p className="text-gray-400 text-sm">{t('auth.subtitle')}</p>
        </div>

        {/* Language toggle */}
        <div className="flex justify-center mb-5 gap-2">
          {availableLocales.map((loc) => (
            <button
              key={loc}
              onClick={() => setLocale(loc)}
              className={`px-3 py-1 rounded text-xs transition-colors ${
                locale === loc
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              {loc === 'en-US' ? 'EN' : 'KR'}
            </button>
          ))}
        </div>

        {/* Card */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 space-y-5">

          {/* ── Server Mode: Status Indicator ── */}
          {isServerMode && (
            <div>
              {serverModeWaiting ? (
                <div className="flex items-center justify-center gap-2 py-3">
                  <Loader size={14} className="text-blue-400 animate-spin" />
                  <span className="text-sm text-gray-400">{t('server.login_starting')}</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-xs text-green-400">
                  <CheckCircle size={12} />
                  <span>{t('server.login_ready')}</span>
                  <span className="text-gray-600 font-mono ml-auto">localhost:{serverPort || 8000}</span>
                </div>
              )}
            </div>
          )}

          {/* ── Server Connection Section (client/web mode only) ── */}
          {!isServerMode && (
            <div>
              <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Wifi size={12} />
                {t('auth.server_section')}
              </h3>

              {/* mDNS Discovered Servers (Electron only) */}
              {isElectron && (mdnsServers.length > 0 || browsing) && (
                <div className="mb-3">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Radio size={10} className={browsing ? 'text-green-400 animate-pulse' : 'text-gray-500'} />
                    <span className="text-[10px] text-gray-500 uppercase">
                      {t('auth.discovered_servers')}
                    </span>
                  </div>
                  {mdnsServers.length > 0 ? (
                    <div className="space-y-1">
                      {mdnsServers.map((s) => {
                        const addr = s.addresses?.[0] || s.host;
                        const sUrl = `http://${addr}:${s.port}`;
                        const isSelected = serverUrl === sUrl;
                        return (
                          <button
                            key={s.name}
                            type="button"
                            onClick={() => handleSelectMdns(s)}
                            className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-left text-sm transition-colors ${
                              isSelected
                                ? 'bg-blue-900/40 border border-blue-600'
                                : 'bg-gray-900/50 border border-gray-700 hover:border-gray-500'
                            }`}
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="w-1.5 h-1.5 rounded-full bg-green-400 shrink-0" />
                              <span className="text-white truncate font-medium">
                                {s.serverName || s.name}
                              </span>
                            </div>
                            <span className="text-xs text-gray-500 font-mono shrink-0 ml-2">
                              {addr}:{s.port}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  ) : browsing ? (
                    <p className="text-xs text-gray-600 italic">{t('auth.discovering')}</p>
                  ) : null}
                </div>
              )}

              {/* Recent Servers */}
              {filteredHistory.length > 0 && (
                <div className="mb-3">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Clock size={10} className="text-gray-500" />
                    <span className="text-[10px] text-gray-500 uppercase">
                      {t('auth.recent_servers')}
                    </span>
                  </div>
                  <div className="space-y-1">
                    {filteredHistory.slice(0, 3).map((h) => {
                      const isSelected = serverUrl === h.url;
                      return (
                        <button
                          key={h.url}
                          type="button"
                          onClick={() => handleSelectServer(h.url)}
                          className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-left text-sm transition-colors group ${
                            isSelected
                              ? 'bg-blue-900/40 border border-blue-600'
                              : 'bg-gray-900/50 border border-gray-700 hover:border-gray-500'
                          }`}
                        >
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <span className="text-white truncate text-sm">
                                {h.name || h.url.replace(/^https?:\/\//, '')}
                              </span>
                            </div>
                            <div className="flex items-center gap-2 mt-0.5">
                              {h.lastUsername && (
                                <span className="text-[10px] text-gray-500">{h.lastUsername}</span>
                              )}
                              <span className="text-[10px] text-gray-600">
                                {formatRelativeTime(h.lastConnected, t)}
                              </span>
                            </div>
                          </div>
                          <span
                            role="button"
                            tabIndex={0}
                            onClick={(e) => handleRemoveHistory(e, h.url)}
                            onKeyDown={(e) => { if (e.key === 'Enter') handleRemoveHistory(e, h.url); }}
                            className="p-1 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0 cursor-pointer"
                            title={t('auth.remove_history')}
                          >
                            <X size={12} />
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Server URL Input */}
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('auth.server_url')}</label>
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Server size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                    <input
                      type="text"
                      value={serverUrl}
                      onChange={(e) => {
                        setServerUrlLocal(e.target.value);
                        setServerStatus(null);
                        setServerName('');
                      }}
                      placeholder="http://192.168.1.10:8000"
                      className="w-full pl-10 pr-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none font-mono"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => handleCheckServer()}
                    disabled={!serverUrl.trim() || serverStatus === 'checking'}
                    className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-xs text-gray-300 hover:bg-gray-600 disabled:opacity-50 transition-colors"
                  >
                    {serverStatus === 'checking' ? '...' : t('auth.check')}
                  </button>
                </div>
                {serverStatus === 'ok' && (
                  <div className="flex items-center gap-1 mt-1 text-xs text-green-400">
                    <CheckCircle size={12} />
                    {serverName
                      ? t('auth.connected_to', { name: serverName })
                      : t('auth.server_connected')}
                  </div>
                )}
                {serverStatus === 'error' && (
                  <div className="flex items-center gap-1 mt-1 text-xs text-red-400">
                    <XCircle size={12} /> {serverError}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── Divider ── */}
          {!serverModeWaiting && <div className="border-t border-gray-700" />}

          {/* ── Login / Register Section ── */}
          {!serverModeWaiting && (
            <div>
              {/* Tab toggle */}
              <div className="flex mb-4 bg-gray-900 rounded-lg p-1">
                <button
                  onClick={() => setMode('login')}
                  className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                    mode === 'login'
                      ? 'bg-gray-700 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <LogIn size={16} />
                  {t('auth.login')}
                </button>
                <button
                  onClick={() => setMode('register')}
                  className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                    mode === 'register'
                      ? 'bg-gray-700 text-white'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <UserPlus size={16} />
                  {t('auth.register')}
                </button>
              </div>

              <form onSubmit={handleSubmit} className="space-y-4">
                {/* Invite Code (register only) */}
                {mode === 'register' && (
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">{t('auth.invite_code')}</label>
                    <input
                      type="text"
                      value={inviteCode}
                      onChange={(e) => setInviteCode(e.target.value)}
                      placeholder={t('auth.invite_code_placeholder')}
                      className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                      required
                    />
                  </div>
                )}

                {/* Username */}
                <div>
                  <label className="block text-xs text-gray-400 mb-1">{t('auth.username')}</label>
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder={t('auth.username_placeholder')}
                    className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                    required
                  />
                </div>

                {/* Email (optional in register) */}
                {mode === 'register' && (
                  <div>
                    <label className="block text-xs text-gray-400 mb-1">
                      {t('auth.email')} <span className="text-gray-600">({t('label.optional')})</span>
                    </label>
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="user@example.com"
                      className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                )}

                {/* Password */}
                <div>
                  <label className="block text-xs text-gray-400 mb-1">{t('auth.password')}</label>
                  <div className="relative">
                    <input
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="••••••••"
                      className="w-full px-3 py-2 pr-10 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                      required
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300"
                    >
                      {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>

                {/* Error */}
                {error && (
                  <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400">
                    {error}
                  </div>
                )}

                {/* Submit */}
                <button
                  type="submit"
                  disabled={submitting || !serverUrl.trim()}
                  className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium transition-colors"
                >
                  {submitting
                    ? '...'
                    : mode === 'login'
                      ? t('auth.login')
                      : t('auth.register')}
                </button>
              </form>

              {/* Server mode: default credentials hint */}
              {isServerMode && mode === 'login' && (
                <p className="text-xs text-gray-500 mt-3 text-center">
                  {t('server.login_hint')}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Desktop App Download */}
        {onShowDownload && (
          <button
            onClick={onShowDownload}
            className="mt-4 w-full p-3 bg-gray-800/50 border border-emerald-700/30 rounded-lg hover:bg-emerald-900/20 transition-colors group"
          >
            <div className="flex items-center justify-center gap-2 text-emerald-400 text-sm group-hover:text-emerald-300">
              <Download size={14} />
              <span className="font-medium">{t('download.get_app')}</span>
            </div>
            <p className="text-xs text-gray-500 mt-1.5 text-center">{t('download.banner_desc')}</p>
          </button>
        )}

        {/* Footer */}
        <p className="text-center text-xs text-gray-600 mt-4">
          Imagine Server v4.0
        </p>
      </div>
    </div>
  );
}
