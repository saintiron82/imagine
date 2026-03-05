/**
 * LoginPage — unified first screen.
 *
 * Features:
 * - Login / Register (always available when server initialized)
 * - "Create Group" panel (always visible, for first-time or re-init)
 * - Remote server connection (Electron only)
 * - mDNS auto-discovery
 * - Tier selection + environment check (during group creation)
 * - Language switcher + quit button
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { setServerUrl as setClientServerUrl, getServerUrl } from '../api/client';
import { useMdnsDiscovery } from '../hooks/useMdnsDiscovery';
import { getServerInfo, initServer, getMe } from '../api/auth';
import {
  getServerHistory,
  addServerToHistory,
  removeServerFromHistory,
  formatRelativeTime,
} from '../utils/serverHistory';
import {
  LogIn, UserPlus, Server, Eye, EyeOff, CheckCircle, XCircle,
  Download, Wifi, X, Clock, Radio, Loader, Users, Languages,
  Plus, ArrowLeft, ArrowRight, Zap, Star, Rocket, AlertCircle,
  Loader2, SkipForward, Lock, User, Globe,
} from 'lucide-react';

const TIERS = [
  { id: 'standard', icon: Zap, color: 'emerald' },
  { id: 'pro', icon: Star, color: 'blue' },
  { id: 'ultra', icon: Rocket, color: 'purple' },
];

export default function LoginPage({ onShowDownload, onLoginComplete, serverRunning, serverPort }) {
  const { login, register, error, checkServerHealth } = useAuth();
  const { t, locale, setLocale, availableLocales } = useLocale();

  // --- View state ---
  const [view, setView] = useState('main'); // 'main' | 'createGroup' | 'tier' | 'checking' | 'install' | 'installing' | 'done' | 'remote'
  const [mode, setMode] = useState('login'); // 'login' | 'register'

  // --- Server state ---
  const [serverUrl, setServerUrlLocal] = useState(
    localStorage.getItem('imagine-server-url') || `http://localhost:${serverPort || 8000}`
  );
  const [serverStatus, setServerStatus] = useState(null);
  const [serverError, setServerError] = useState('');
  const [serverName, setServerName] = useState('');
  const [groupName, setGroupName] = useState('');
  const [serverInitialized, setServerInitialized] = useState(null); // null=unknown, true/false

  // --- Login/Register form ---
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [serverPassword, setServerPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // --- Create Group form ---
  const [createGroupName, setCreateGroupName] = useState('');
  const [createServerPassword, setCreateServerPassword] = useState('');
  const [createAdminUsername, setCreateAdminUsername] = useState('');
  const [createAdminPassword, setCreateAdminPassword] = useState('');
  const [createError, setCreateError] = useState('');
  const [createSubmitting, setCreateSubmitting] = useState(false);
  const [selectedTier, setSelectedTier] = useState(null);

  // --- Environment check ---
  const [envStatus, setEnvStatus] = useState(null);
  const [installLogs, setInstallLogs] = useState([]);
  const [installDone, setInstallDone] = useState(false);
  const [installSuccess, setInstallSuccess] = useState(false);
  const logEndRef = useRef(null);

  // --- Server history + mDNS ---
  const [history, setHistory] = useState([]);
  const { servers: mdnsServers, browsing } = useMdnsDiscovery();

  // --- Auto-detect server state on mount ---
  const healthCheckDone = useRef(false);

  useEffect(() => {
    if (!serverRunning) return;
    if (healthCheckDone.current) return;
    healthCheckDone.current = true;

    const checkServer = async () => {
      const url = `http://localhost:${serverPort || 8000}`;
      setServerUrlLocal(url);
      setClientServerUrl(url);

      try {
        const info = await getServerInfo(url);
        if (info.ok) {
          setServerInitialized(info.initialized);
          if (info.group_name) setGroupName(info.group_name);
          setServerStatus('ok');
          // Auto-fill username from history
          const entry = getServerHistory().find(h => h.url === url);
          if (entry?.lastUsername && !username) setUsername(entry.lastUsername);
        }
      } catch {
        // Server not ready yet
      }
    };

    const timer = setTimeout(checkServer, 500);
    return () => clearTimeout(timer);
  }, [serverRunning, serverPort]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load history for remote view
  useEffect(() => {
    setHistory(getServerHistory());
  }, []);

  // Auto-scroll install logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [installLogs]);

  // Listen for install-log events
  useEffect(() => {
    const pipeline = window.electron?.pipeline;
    if (!pipeline) return;
    const handler = (data) => {
      setInstallLogs(prev => [...prev, data]);
      if (data.done) {
        setInstallDone(true);
        setInstallSuccess(data.type === 'success');
        setView('done');
      }
    };
    pipeline.onInstallLog?.(handler);
    return () => pipeline.offInstallLog?.();
  }, []);

  // --- Handlers ---

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
      const info = await getServerInfo(targetUrl);
      if (info.ok) {
        setServerInitialized(info.initialized);
        if (info.group_name) setGroupName(info.group_name);
      }
    } else {
      setServerStatus('error');
      setServerError(result.error);
    }
  };

  const handleSelectServer = (url) => {
    setServerUrlLocal(url);
    setServerStatus(null);
    handleCheckServer(url);
    const entry = getServerHistory().find(h => h.url === url);
    if (entry?.lastUsername) setUsername(entry.lastUsername);
  };

  const handleSelectMdns = (server) => {
    const addr = server.addresses?.[0] || server.host;
    const url = `http://${addr}:${server.port}`;
    setServerUrlLocal(url);
    setServerStatus(null);
    handleCheckServer(url);
    const entry = getServerHistory().find(h => h.url === url);
    if (entry?.lastUsername) setUsername(entry.lastUsername);
  };

  const handleRemoveHistory = (e, url) => {
    e.stopPropagation();
    removeServerFromHistory(url);
    setHistory(getServerHistory());
  };

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);

    const trimmedUrl = serverUrl.trim();
    let success;

    if (mode === 'login') {
      success = await login({ username, password, serverUrl: trimmedUrl });
    } else {
      success = await register({
        server_password: serverPassword,
        username,
        email: email || undefined,
        password,
        serverUrl: trimmedUrl,
      });
    }

    if (success) {
      addServerToHistory({
        url: trimmedUrl,
        name: serverName || groupName,
        version: '',
        lastUsername: username,
      });

      // Determine mode: check if connecting to local server → use role
      // Remote server → always client mode
      const isLocal = trimmedUrl.includes('localhost') || trimmedUrl.includes('127.0.0.1');
      if (onLoginComplete) {
        try {
          const me = await getMe();
          const user = me.user || me;
          if (isLocal && user.role === 'admin') {
            onLoginComplete('server');
          } else {
            onLoginComplete('client');
          }
        } catch {
          onLoginComplete('client');
        }
      }
    }

    setSubmitting(false);
  };

  // --- Create Group Handlers ---

  const createChecks = {
    groupName: createGroupName.trim().length > 0,
    serverPassword: createServerPassword.trim().length >= 1,
    adminUsername: createAdminUsername.trim().length >= 2,
    adminPassword: createAdminPassword.length >= 4,
  };
  const canCreate = createChecks.groupName && createChecks.serverPassword && createChecks.adminUsername && createChecks.adminPassword;
  const createTouched = {
    groupName: createGroupName.length > 0,
    serverPassword: createServerPassword.length > 0,
    adminUsername: createAdminUsername.length > 0,
    adminPassword: createAdminPassword.length > 0,
  };

  const handleCreateNext = async () => {
    if (!canCreate) return;
    setCreateError('');
    setCreateSubmitting(true);

    try {
      const port = serverPort || 8000;

      // 1. Electron: ensure server is running
      if (isElectron) {
        const status = await window.electron?.server?.getStatus();
        if (!status?.running) {
          await window.electron.server.start({ port });
          await new Promise(r => setTimeout(r, 1500));
        }
      }

      // 2. Verify server is reachable + check initialization status
      const baseUrl = `http://localhost:${port}`;
      const info = await getServerInfo(baseUrl);
      if (info.ok && info.initialized) {
        setCreateError(t('validation.server_already_initialized'));
        setCreateSubmitting(false);
        return;
      }
      if (!info.ok) {
        setCreateError(t('auth.server_unreachable') || 'Server connection failed');
        setCreateSubmitting(false);
        return;
      }

      // 3. Initialize server NOW (creates group + admin)
      setClientServerUrl(baseUrl);
      await initServer(baseUrl, {
        group_name: createGroupName,
        server_password: createServerPassword,
        admin_username: createAdminUsername,
        admin_password: createAdminPassword,
      });

      // 4. Server init succeeded — proceed to tier or finish
      setCreateSubmitting(false);
      if (isElectron && window.electron?.pipeline?.checkEnv) {
        setView('tier');
      } else {
        if (onLoginComplete) onLoginComplete('server');
      }
    } catch (e) {
      setCreateError(e.message || 'Server initialization failed');
      setCreateSubmitting(false);
    }
  };

  const handleTierConfirm = useCallback(async () => {
    if (!selectedTier) return;

    try {
      await window.electron.pipeline.updateConfig('ai_mode.override', selectedTier);
      await window.electron.pipeline.updateConfig('ai_mode.auto_detect', false);
    } catch (e) {
      console.error('Failed to save tier:', e);
    }

    setView('checking');
    try {
      const status = await window.electron.pipeline.checkEnv();
      setEnvStatus(status);
      const modelsOk = status.visual_model_cached && status.dependencies_ok;
      if (modelsOk) {
        // Init already done — just complete login
        if (onLoginComplete) onLoginComplete('server');
      } else {
        setView('install');
      }
    } catch (e) {
      console.error('check-env failed:', e);
      if (onLoginComplete) onLoginComplete('server');
    }
  }, [selectedTier, onLoginComplete]);

  const handleInstall = useCallback(() => {
    setView('installing');
    setInstallLogs([]);
    window.electron?.pipeline?.installEnv();
  }, []);

  const handleSkip = useCallback(() => {
    if (onLoginComplete) onLoginComplete('server');
  }, [onLoginComplete]);

  const handleFinish = useCallback(() => {
    if (onLoginComplete) onLoginComplete('server');
  }, [onLoginComplete]);

  // --- Filter history ---
  const mdnsUrls = new Set(
    mdnsServers.map((s) => `http://${s.addresses?.[0]}:${s.port}`)
  );
  const filteredHistory = history.filter((h) => !mdnsUrls.has(h.url));

  // --- Shared top bar ---
  const renderTopBar = () => (
    <div
      className="fixed top-0 left-0 right-0 h-10 flex items-center justify-between px-4 z-50"
      style={isElectron ? { WebkitAppRegion: 'drag' } : {}}
    >
      <div className="flex items-center gap-2" style={{ WebkitAppRegion: 'no-drag' }}>
        {groupName && (
          <span className="text-xs text-blue-300 bg-blue-900/30 px-2 py-0.5 rounded flex items-center gap-1">
            <Users size={10} />
            {groupName}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2" style={{ WebkitAppRegion: 'no-drag' }}>
        {/* Language toggle */}
        <button
          onClick={() => {
            const idx = availableLocales.indexOf(locale);
            setLocale(availableLocales[(idx + 1) % availableLocales.length]);
          }}
          className="p-1.5 text-gray-500 hover:text-white transition-colors"
          title="Language"
        >
          <Languages size={14} />
        </button>
        {/* Quit button (Electron only) */}
        {isElectron && (
          <button
            onClick={() => window.electron?.app?.quit()}
            className="p-1.5 text-gray-500 hover:text-red-400 transition-colors"
            title={t('action.quit')}
          >
            <X size={14} />
          </button>
        )}
      </div>
    </div>
  );

  // ============================================================
  // TIER SELECTION VIEW
  // ============================================================
  if (view === 'tier') {
    const colorMap = {
      emerald: { active: 'border-emerald-500 bg-emerald-900/20 shadow-lg shadow-emerald-900/30', icon: 'bg-emerald-600', dot: 'bg-emerald-400' },
      blue: { active: 'border-blue-500 bg-blue-900/20 shadow-lg shadow-blue-900/30', icon: 'bg-blue-600', dot: 'bg-blue-400' },
      purple: { active: 'border-purple-500 bg-purple-900/20 shadow-lg shadow-purple-900/30', icon: 'bg-purple-600', dot: 'bg-purple-400' },
    };
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        {renderTopBar()}
        <div className="max-w-3xl w-full px-8">
          <div className="text-center mb-10">
            <h1 className="text-2xl font-bold mb-2">{t('setup.select_tier')}</h1>
            <p className="text-gray-400 text-sm">{t('setup.select_tier_desc')}</p>
          </div>
          <div className="grid grid-cols-3 gap-4 mb-8">
            {TIERS.map(({ id, icon: Icon, color }) => {
              const isSelected = selectedTier === id;
              const cm = colorMap[color];
              return (
                <button key={id} onClick={() => setSelectedTier(id)}
                  className={`p-5 rounded-xl border-2 text-left transition-all ${
                    isSelected ? cm.active : 'border-gray-700 bg-gray-800/50 hover:border-gray-500'
                  }`}>
                  <div className="flex items-center gap-3 mb-3">
                    <div className={`p-2 rounded-lg ${isSelected ? cm.icon : 'bg-gray-700'}`}>
                      <Icon size={20} />
                    </div>
                    <h2 className="text-base font-bold">{t(`setup.tier_${id}_title`)}</h2>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed mb-3">{t(`setup.tier_${id}_desc`)}</p>
                  <div className="space-y-1.5 text-xs text-gray-500">
                    <div className="flex items-center gap-1.5">
                      <span className={`w-1 h-1 rounded-full ${cm.dot}`} />
                      {t(`setup.tier_${id}_vram`)}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className={`w-1 h-1 rounded-full ${cm.dot}`} />
                      VLM: {t(`setup.tier_${id}_vlm`)}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
          <div className="flex justify-between items-center">
            <button onClick={() => setView('createGroup')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white transition-colors">
              <ArrowLeft size={16} />
              {t('action.back')}
            </button>
            <div className="flex items-center gap-4">
              <p className="text-xs text-gray-600">{t('setup.changeable_later')}</p>
              <button onClick={handleTierConfirm} disabled={!selectedTier}
                className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                  selectedTier ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                }`}>
                {t('setup.next')}
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================
  // CHECKING ENVIRONMENT VIEW
  // ============================================================
  if (view === 'checking') {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        {renderTopBar()}
        <div className="text-center max-w-md">
          {createError ? (
            <>
              <AlertCircle size={40} className="text-red-400 mx-auto mb-4" />
              <p className="text-gray-300 mb-3">{t('setup.checking_env')}</p>
              <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400 mb-4">{createError}</div>
              <button onClick={() => { setCreateError(''); setView('createGroup'); }}
                className="px-4 py-2 rounded-lg text-sm text-gray-400 hover:text-white bg-gray-800 hover:bg-gray-700 transition-colors">
                <ArrowLeft size={14} className="inline mr-1" />
                {t('action.back')}
              </button>
            </>
          ) : (
            <>
              <Loader2 size={40} className="animate-spin text-blue-400 mx-auto mb-4" />
              <p className="text-gray-300">{t('setup.checking_env')}</p>
            </>
          )}
        </div>
      </div>
    );
  }

  // ============================================================
  // INSTALL MODELS VIEW
  // ============================================================
  if (view === 'install') {
    const missing = [];
    if (!envStatus?.visual_model_cached) missing.push('SigLIP2 (VV)');
    if (!envStatus?.dependencies_ok) missing.push(t('setup.dependencies'));
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        {renderTopBar()}
        <div className="max-w-lg w-full px-8">
          <div className="text-center mb-8">
            <AlertCircle size={48} className="text-yellow-400 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-2">{t('setup.models_required')}</h2>
            <p className="text-sm text-gray-400">{t('setup.models_required_desc')}</p>
          </div>
          {missing.length > 0 && (
            <div className="bg-gray-800 rounded-lg p-4 mb-6 border border-gray-700">
              <p className="text-xs text-gray-500 mb-2">{t('setup.missing_items')}</p>
              {missing.map((item, i) => (
                <div key={i} className="flex items-center gap-2 text-sm text-yellow-300">
                  <AlertCircle size={14} /> {item}
                </div>
              ))}
            </div>
          )}
          <div className="flex gap-3">
            <button onClick={handleInstall}
              className="flex-1 flex items-center justify-center gap-2 px-6 py-3 rounded-lg font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors">
              <Download size={18} /> {t('setup.download_models')}
            </button>
            <button onClick={handleSkip}
              className="flex items-center gap-2 px-4 py-3 rounded-lg font-medium bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors">
              <SkipForward size={16} /> {t('setup.skip')}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================
  // INSTALLING VIEW
  // ============================================================
  if (view === 'installing' || (view === 'done' && installDone)) {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        {renderTopBar()}
        <div className="max-w-2xl w-full px-8">
          <div className="text-center mb-6">
            {view === 'installing' ? (
              <>
                <Loader2 size={40} className="animate-spin text-blue-400 mx-auto mb-4" />
                <h2 className="text-xl font-bold mb-1">{t('setup.downloading')}</h2>
                <p className="text-xs text-gray-500">{t('setup.downloading_desc')}</p>
              </>
            ) : installSuccess ? (
              <>
                <CheckCircle size={40} className="text-green-400 mx-auto mb-4" />
                <h2 className="text-xl font-bold mb-1">{t('setup.install_complete')}</h2>
              </>
            ) : (
              <>
                <AlertCircle size={40} className="text-red-400 mx-auto mb-4" />
                <h2 className="text-xl font-bold mb-1">{t('setup.install_failed')}</h2>
                <p className="text-xs text-gray-500">{t('setup.install_failed_desc')}</p>
              </>
            )}
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 mb-6">
            <div className="h-64 overflow-y-auto p-3 font-mono text-xs space-y-0.5">
              {installLogs.map((log, i) => (
                <div key={i} className={
                  log.type === 'error' ? 'text-red-400' :
                  log.type === 'success' ? 'text-green-400' :
                  log.type === 'warning' ? 'text-yellow-400' : 'text-gray-400'
                }>{log.message}</div>
              ))}
              <div ref={logEndRef} />
            </div>
          </div>
          {view === 'done' && (
            <div className="text-center">
              <button onClick={handleFinish} disabled={createSubmitting}
                className="flex items-center gap-2 mx-auto px-6 py-3 rounded-lg font-medium bg-blue-600 hover:bg-blue-500 text-white transition-colors disabled:opacity-50">
                {createSubmitting ? <Loader2 size={16} className="animate-spin" /> : null}
                {t('setup.start')}
                <ArrowRight size={16} />
              </button>
              {createError && <p className="text-xs text-red-400 mt-2">{createError}</p>}
            </div>
          )}
        </div>
      </div>
    );
  }

  // ============================================================
  // CREATE GROUP VIEW
  // ============================================================
  if (view === 'createGroup') {
    const fieldClass = (key) => {
      if (!createTouched[key]) return 'border-gray-600';
      return createChecks[key] ? 'border-green-600' : 'border-red-600';
    };

    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        {renderTopBar()}
        <div className="max-w-md w-full px-8">
          <div className="text-center mb-8">
            <div className="p-3 rounded-full bg-blue-600/20 w-fit mx-auto mb-4">
              <Plus size={28} className="text-blue-400" />
            </div>
            <h1 className="text-2xl font-bold mb-2">{t('group.create_title')}</h1>
            <p className="text-gray-400 text-sm">{t('group.create_desc')}</p>
          </div>

          <div className="space-y-4">
            {/* Group Name */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('group.name')}</label>
              <input type="text" value={createGroupName} onChange={(e) => setCreateGroupName(e.target.value)}
                placeholder={t('group.name_placeholder')}
                className={`w-full px-3 py-2 bg-gray-900 border rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none ${fieldClass('groupName')}`} />
              {createTouched.groupName && !createChecks.groupName && (
                <p className="text-[10px] text-red-400 mt-1">{t('validation.group_name_required')}</p>
              )}
            </div>

            {/* Server Password */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('group.server_password')}</label>
              <input type="password" value={createServerPassword} onChange={(e) => setCreateServerPassword(e.target.value)}
                placeholder={t('group.server_password_placeholder')}
                className={`w-full px-3 py-2 bg-gray-900 border rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none ${fieldClass('serverPassword')}`} />
              <p className="text-[10px] text-gray-600 mt-1">{t('group.server_password_hint')}</p>
            </div>

            {/* Admin Username */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('group.admin_username')}</label>
              <input type="text" value={createAdminUsername} onChange={(e) => setCreateAdminUsername(e.target.value)}
                placeholder={t('group.admin_username_placeholder')}
                className={`w-full px-3 py-2 bg-gray-900 border rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none ${fieldClass('adminUsername')}`} />
              {createTouched.adminUsername && !createChecks.adminUsername && (
                <p className="text-[10px] text-red-400 mt-1">{t('validation.username_min')}</p>
              )}
            </div>

            {/* Admin Password */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('group.admin_password')}</label>
              <input type="password" value={createAdminPassword} onChange={(e) => setCreateAdminPassword(e.target.value)}
                placeholder="••••"
                className={`w-full px-3 py-2 bg-gray-900 border rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none ${fieldClass('adminPassword')}`} />
              {createTouched.adminPassword && !createChecks.adminPassword && (
                <p className="text-[10px] text-red-400 mt-1">{t('validation.password_min_4')}</p>
              )}
            </div>

            {/* Validation summary */}
            <div className="flex gap-3 text-[10px] text-gray-500 pt-1">
              {Object.entries(createChecks).map(([key, ok]) => (
                <span key={key} className={ok ? 'text-green-500' : 'text-gray-600'}>
                  {ok ? '✓' : '○'} {t(`validation.${key === 'groupName' ? 'group_name_required' : key === 'serverPassword' ? 'server_password_required' : key === 'adminUsername' ? 'username_min' : 'password_min_4'}`).split(' ')[0]}
                </span>
              ))}
            </div>

            {/* Error */}
            {createError && (
              <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400">{createError}</div>
            )}

            {/* Buttons */}
            <div className="flex gap-3 pt-2">
              <button onClick={() => setView('main')}
                className="flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm text-gray-400 hover:text-white transition-colors">
                <ArrowLeft size={16} /> {t('action.back')}
              </button>
              <button onClick={handleCreateNext}
                disabled={!canCreate || createSubmitting}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  canCreate && !createSubmitting ? 'bg-blue-600 hover:bg-blue-500 text-white' : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                }`}>
                {createSubmitting ? <Loader size={14} className="animate-spin" /> : null}
                {t('setup.next')}
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // ============================================================
  // REMOTE SERVER VIEW (Electron only)
  // ============================================================
  if (view === 'remote') {
    return (
      <div className="flex items-center justify-center h-screen bg-gray-900 text-white">
        {renderTopBar()}
        <div className="w-full max-w-md px-8">
          <div className="text-center mb-6">
            <Globe size={28} className="text-emerald-400 mx-auto mb-3" />
            <h1 className="text-2xl font-bold mb-2">{t('auth.server_section')}</h1>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-6 space-y-4">
            {/* mDNS */}
            {isElectron && (mdnsServers.length > 0 || browsing) && (
              <div>
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Radio size={10} className={browsing ? 'text-green-400 animate-pulse' : 'text-gray-500'} />
                  <span className="text-[10px] text-gray-500 uppercase">{t('auth.discovered_servers')}</span>
                </div>
                {mdnsServers.length > 0 ? (
                  <div className="space-y-1">
                    {mdnsServers.map((s) => {
                      const addr = s.addresses?.[0] || s.host;
                      const sUrl = `http://${addr}:${s.port}`;
                      const isSelected = serverUrl === sUrl;
                      return (
                        <button key={s.name} type="button" onClick={() => handleSelectMdns(s)}
                          className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-left text-sm transition-colors ${
                            isSelected ? 'bg-blue-900/40 border border-blue-600' : 'bg-gray-900/50 border border-gray-700 hover:border-gray-500'
                          }`}>
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-400 shrink-0" />
                            <span className="text-white truncate font-medium">{s.serverName || s.name}</span>
                          </div>
                          <span className="text-xs text-gray-500 font-mono shrink-0 ml-2">{addr}:{s.port}</span>
                        </button>
                      );
                    })}
                  </div>
                ) : browsing ? (
                  <p className="text-xs text-gray-600 italic">{t('auth.discovering')}</p>
                ) : null}
              </div>
            )}

            {/* History */}
            {filteredHistory.length > 0 && (
              <div>
                <div className="flex items-center gap-1.5 mb-1.5">
                  <Clock size={10} className="text-gray-500" />
                  <span className="text-[10px] text-gray-500 uppercase">{t('auth.recent_servers')}</span>
                </div>
                <div className="space-y-1">
                  {filteredHistory.slice(0, 3).map((h) => {
                    const isSelected = serverUrl === h.url;
                    return (
                      <button key={h.url} type="button" onClick={() => handleSelectServer(h.url)}
                        className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-left text-sm transition-colors group ${
                          isSelected ? 'bg-blue-900/40 border border-blue-600' : 'bg-gray-900/50 border border-gray-700 hover:border-gray-500'
                        }`}>
                        <div className="min-w-0 flex-1">
                          <span className="text-white truncate text-sm">{h.name || h.url.replace(/^https?:\/\//, '')}</span>
                          <div className="flex items-center gap-2 mt-0.5">
                            {h.lastUsername && <span className="text-[10px] text-gray-500">{h.lastUsername}</span>}
                            <span className="text-[10px] text-gray-600">{formatRelativeTime(h.lastConnected, t)}</span>
                          </div>
                        </div>
                        <span role="button" tabIndex={0}
                          onClick={(e) => handleRemoveHistory(e, h.url)}
                          onKeyDown={(e) => { if (e.key === 'Enter') handleRemoveHistory(e, h.url); }}
                          className="p-1 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0 cursor-pointer">
                          <X size={12} />
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* URL Input */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('auth.server_url')}</label>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Server size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                  <input type="text" value={serverUrl}
                    onChange={(e) => { setServerUrlLocal(e.target.value); setServerStatus(null); }}
                    placeholder="http://192.168.1.10:8000"
                    className="w-full pl-10 pr-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none font-mono" />
                </div>
                <button type="button" onClick={() => handleCheckServer()}
                  disabled={!serverUrl.trim() || serverStatus === 'checking'}
                  className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-xs text-gray-300 hover:bg-gray-600 disabled:opacity-50 transition-colors">
                  {serverStatus === 'checking' ? '...' : t('auth.check')}
                </button>
              </div>
              {serverStatus === 'ok' && (
                <div className="flex items-center gap-1 mt-1 text-xs text-green-400">
                  <CheckCircle size={12} />
                  {groupName ? t('auth.connected_to', { name: groupName }) : t('auth.server_connected')}
                </div>
              )}
              {serverStatus === 'error' && (
                <div className="flex items-center gap-1 mt-1 text-xs text-red-400">
                  <XCircle size={12} /> {serverError}
                </div>
              )}
            </div>

            {/* Connect button → go to main with remote URL */}
            {serverStatus === 'ok' && (
              <button onClick={() => { localStorage.setItem('imagine-server-url', serverUrl.trim()); setView('main'); }}
                className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg text-sm font-medium transition-colors">
                {t('auth.connect')}
              </button>
            )}
          </div>

          <button onClick={() => setView('main')}
            className="mt-4 flex items-center gap-2 mx-auto px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors">
            <ArrowLeft size={16} /> {t('action.back')}
          </button>
        </div>
      </div>
    );
  }

  // ============================================================
  // MAIN VIEW — Login/Register + Side actions
  // ============================================================
  const serverWaiting = isElectron && !serverRunning;

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900">
      {renderTopBar()}
      <div className="w-full max-w-md p-8 pt-14">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-white mb-2">Imagine</h1>
          <p className="text-gray-400 text-sm">{t('auth.subtitle')}</p>
        </div>

        {/* Server waiting */}
        {serverWaiting && (
          <div className="flex items-center justify-center gap-2 py-3 mb-4">
            <Loader size={14} className="text-blue-400 animate-spin" />
            <span className="text-sm text-gray-400">{t('server.login_starting')}</span>
          </div>
        )}

        {/* === Login Card === */}
        <div className={`bg-gray-800 rounded-lg border border-gray-700 p-6 ${
          serverWaiting ? 'opacity-50 pointer-events-none' : ''
        }`}>
          {/* Top action buttons */}
          <div className="flex items-center gap-2 mb-4">
            {/* Group name badge */}
            {groupName && serverInitialized && (
              <span className="flex items-center gap-1.5 text-xs text-blue-300 bg-blue-900/20 border border-blue-800/40 px-2.5 py-1 rounded">
                <Users size={12} className="text-blue-400" />
                {groupName}
              </span>
            )}
            <div className="flex-1" />
            {/* Create Group — always visible */}
            <button onClick={() => setView('createGroup')}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                serverInitialized === false
                  ? 'bg-blue-600 text-white hover:bg-blue-500'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}>
              <Plus size={12} />
              {t('group.create_title')}
            </button>
            {/* Remote Server — Electron only */}
            {isElectron && (
              <button onClick={() => setView('remote')}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs text-gray-500 hover:text-white hover:bg-gray-700 transition-colors">
                <Globe size={12} />
                {t('auth.remote_server')}
              </button>
            )}
          </div>

          {/* Server not initialized hint */}
          {serverInitialized === false && !serverWaiting && (
            <div className="p-3 mb-4 bg-yellow-900/20 border border-yellow-800/40 rounded-lg text-xs text-yellow-300">
              {t('group.server_not_initialized')}
            </div>
          )}

          {/* Tab toggle */}
          <div className={`flex mb-4 bg-gray-900 rounded-lg p-1 ${serverInitialized === false ? 'opacity-40 pointer-events-none' : ''}`}>
            <button onClick={() => setMode('login')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                mode === 'login' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'
              }`}>
              <LogIn size={16} /> {t('auth.login')}
            </button>
            <button onClick={() => setMode('register')}
              className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                mode === 'register' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:text-white'
              }`}>
              <UserPlus size={16} /> {t('auth.register')}
            </button>
          </div>

          <form onSubmit={handleLoginSubmit} className={`space-y-4 ${serverInitialized === false ? 'opacity-40 pointer-events-none' : ''}`}>
            {/* Server Password (register only) */}
            {mode === 'register' && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('auth.server_password')}</label>
                <input type="password" value={serverPassword}
                  onChange={(e) => setServerPassword(e.target.value)}
                  placeholder={t('auth.server_password_placeholder')}
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                  required />
                <p className="text-[10px] text-gray-600 mt-1">{t('auth.server_password_hint')}</p>
              </div>
            )}

            {/* Username */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('auth.username')}</label>
              <input type="text" value={username} onChange={(e) => setUsername(e.target.value)}
                placeholder={t('auth.username_placeholder')}
                className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                required />
            </div>

            {/* Email (register only) */}
            {mode === 'register' && (
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  {t('auth.email')} <span className="text-gray-600">({t('label.optional')})</span>
                </label>
                <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                  placeholder="user@example.com"
                  className="w-full px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none" />
              </div>
            )}

            {/* Password */}
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('auth.password')}</label>
              <div className="relative">
                <input type={showPassword ? 'text' : 'password'} value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full px-3 py-2 pr-10 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                  required />
                <button type="button" onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300">
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="p-3 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400">{error}</div>
            )}

            {/* Submit */}
            <button type="submit" disabled={submitting || !serverUrl.trim()}
              className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium transition-colors">
              {submitting ? '...' : mode === 'login' ? t('auth.login') : t('auth.register')}
            </button>
          </form>
        </div>

        {/* Desktop App Download (web only) */}
        {onShowDownload && (
          <button onClick={onShowDownload}
            className="mt-4 w-full p-3 bg-gray-800/50 border border-emerald-700/30 rounded-lg hover:bg-emerald-900/20 transition-colors group">
            <div className="flex items-center justify-center gap-2 text-emerald-400 text-sm group-hover:text-emerald-300">
              <Download size={14} />
              <span className="font-medium">{t('download.get_app')}</span>
            </div>
            <p className="text-xs text-gray-500 mt-1.5 text-center">{t('download.banner_desc')}</p>
          </button>
        )}

        {/* Footer */}
        <p className="text-center text-xs text-gray-600 mt-4">Imagine Server v4.0</p>
      </div>
    </div>
  );
}
