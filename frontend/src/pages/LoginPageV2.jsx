/**
 * LoginPageV2 — Unified login screen (v2 redesign).
 *
 * Design principles:
 * - Always shown first on app launch
 * - Single screen for login / register / create server
 * - Server name → Firebase address resolution (no raw IP input)
 * - Local server auto-detected and pre-filled
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLocale } from '../i18n';
import { isElectron, setServerUrl as setClientServerUrl, getServerUrl } from '../api/client';
import { lookupGroup } from '../api/firebase';
import { getServerInfo, initServer, resetGroup } from '../api/auth';
import { resetDatabase } from '../api/admin';
import {
  LogIn, UserPlus, Plus, Eye, EyeOff, Loader2, ArrowLeft,
  Zap, Star, Rocket, Languages, AlertCircle, CheckCircle, Server,
  AlertTriangle, Trash2, RefreshCw,
} from 'lucide-react';

const TIERS = [
  { id: 'standard', icon: Zap, color: 'emerald' },
  { id: 'pro', icon: Star, color: 'blue' },
  { id: 'ultra', icon: Rocket, color: 'purple' },
];

// ---------------------------------------------------------------------------
// Shared UI — defined outside component to avoid remount on every render
// ---------------------------------------------------------------------------

function InputField({ icon: Icon, label, type = 'text', value, onChange, placeholder, autoFocus, disabled, showPassword, onTogglePassword }) {
  const isPassword = type === 'password';
  const actualType = isPassword && showPassword ? 'text' : type;
  return (
    <div>
      {label && <label className="block text-xs font-medium text-zinc-400 mb-1.5">{label}</label>}
      <div className="relative">
        {Icon && <Icon size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />}
        <input
          type={actualType}
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          autoFocus={autoFocus}
          disabled={disabled}
          className={`w-full ${Icon ? 'pl-9' : 'pl-3'} pr-3 py-2.5 bg-zinc-800/60 border border-zinc-700/60 rounded-lg
            text-sm text-zinc-100 placeholder-zinc-500
            focus:outline-none focus:border-blue-500/60 focus:ring-1 focus:ring-blue-500/30
            disabled:opacity-50 disabled:cursor-not-allowed
            transition-colors`}
        />
        {isPassword && onTogglePassword && (
          <button
            type="button"
            onClick={onTogglePassword}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300"
          >
            {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
          </button>
        )}
      </div>
    </div>
  );
}

function SubmitButton({ children, loading, disabled, resolving, resolvingText }) {
  return (
    <button
      type="submit"
      disabled={disabled || loading}
      className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500
        text-white text-sm font-medium rounded-lg transition-colors
        flex items-center justify-center gap-2"
    >
      {loading ? <Loader2 size={16} className="animate-spin" /> : null}
      {resolving ? resolvingText : children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Resolve a server name to an actual URL. */
async function resolveServer(serverName, localGroupName, localPort) {
  const trimmed = serverName.trim();
  if (!trimmed) return { ok: false, error: 'empty' };

  // 1) If it matches the local server name, use localhost
  if (localGroupName && trimmed.toLowerCase() === localGroupName.toLowerCase()) {
    const url = `http://localhost:${localPort || 8000}`;
    return { ok: true, url, local: true };
  }

  // 2) Firebase lookup
  try {
    const result = await lookupGroup(trimmed);
    if (!result) return { ok: false, error: 'not_found' };

    // Try multiple addresses in order
    const candidates = [];
    if (result.lan_ip) candidates.push(`http://${result.lan_ip}:${result.port}`);
    if (result.public_ip) candidates.push(`http://${result.public_ip}:${result.port}`);
    if (result.url) candidates.push(result.url);

    for (const url of candidates) {
      try {
        const info = await getServerInfo(url);
        if (info.ok) return { ok: true, url, local: false, groupName: info.group_name };
      } catch { /* try next */ }
    }

    // Return first candidate even if health check failed (might just be slow)
    if (candidates.length > 0) {
      return { ok: true, url: candidates[0], local: false };
    }
    return { ok: false, error: 'unreachable' };
  } catch {
    return { ok: false, error: 'firebase_error' };
  }
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function LoginPageV2({ onLoginComplete, serverPort }) {
  const { login, register, error: authError } = useAuth();
  const { t, locale, setLocale, availableLocales } = useLocale();

  // --- View: 'login' | 'register' | 'create' ---
  const [view, setView] = useState('login');

  // --- Local server detection ---
  const [localGroupName, setLocalGroupName] = useState(null);
  const [localServerReady, setLocalServerReady] = useState(false);
  const [localServerStarting, setLocalServerStarting] = useState(false);

  // --- Remember me ---
  const [rememberMe, setRememberMe] = useState(() => {
    return localStorage.getItem('imagine-remember-me') === 'true';
  });
  const savedLogin = rememberMe ? JSON.parse(localStorage.getItem('imagine-saved-login') || 'null') : null;

  // --- Form fields ---
  const [serverName, setServerName] = useState(savedLogin?.serverName || '');
  const [username, setUsername] = useState(savedLogin?.username || '');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [serverPassword, setServerPassword] = useState(savedLogin?.serverPassword || '');
  const [showPassword, setShowPassword] = useState(false);

  // --- Create server fields ---
  const [newGroupName, setNewGroupName] = useState('');
  const [newServerPassword, setNewServerPassword] = useState('');
  const [adminUsername, setAdminUsername] = useState('');
  const [adminPassword, setAdminPassword] = useState('');
  const [selectedTier, setSelectedTier] = useState('pro');

  // --- Existing server conflict handling ---
  const [existingServerHandled, setExistingServerHandled] = useState(false);
  const [oldServerPassword, setOldServerPassword] = useState('');

  // --- Status ---
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [resolving, setResolving] = useState(false);

  const port = serverPort || 8000;

  // -----------------------------------------------------------------------
  // On mount: detect local server
  // -----------------------------------------------------------------------
  useEffect(() => {
    if (!isElectron) return;

    const detectLocal = async () => {
      try {
        // Check if server is already running
        const status = await window.electron?.server?.getStatus();
        if (status?.running) {
          setLocalServerReady(true);
          setClientServerUrl(`http://localhost:${port}`);
          const info = await getServerInfo(`http://localhost:${port}`);
          if (info.ok && info.group_name) {
            setLocalGroupName(info.group_name);
            // Auto-fill server name if user hasn't typed anything yet
            setServerName(prev => prev || info.group_name);
          }
          return;
        }

        // Check if local DB has a group configured (server not running yet)
        // We start the server to check, then it stays running
        setLocalServerStarting(true);
        await window.electron?.server?.start({ port });
        await new Promise(r => setTimeout(r, 2000));

        const info = await getServerInfo(`http://localhost:${port}`);
        if (info.ok && info.initialized && info.group_name) {
          setLocalGroupName(info.group_name);
          setServerName(prev => prev || info.group_name);
          setLocalServerReady(true);
          setClientServerUrl(`http://localhost:${port}`);
        } else if (info.ok && !info.initialized) {
          // Server exists but no group → ready for creation
          setLocalServerReady(true);
          setClientServerUrl(`http://localhost:${port}`);
        } else {
          // Failed to start — stop it
          try { await window.electron?.server?.stop(); } catch { /* ignore */ }
        }
      } catch (e) {
        console.error('Local server detection failed:', e);
      } finally {
        setLocalServerStarting(false);
      }
    };

    detectLocal();
  }, [port]);

  // -----------------------------------------------------------------------
  // Login handler
  // -----------------------------------------------------------------------
  const handleLogin = useCallback(async (e) => {
    e.preventDefault();
    setError('');

    if (!serverName.trim()) {
      setError(t('login2.server_name_required'));
      return;
    }
    if (!serverPassword.trim()) {
      setError(t('validation.server_password_required'));
      return;
    }
    if (!username.trim()) {
      setError(t('validation.field_required'));
      return;
    }
    if (!password) {
      setError(t('validation.field_required'));
      return;
    }

    setLoading(true);
    setResolving(true);

    try {
      // Resolve server name to URL
      const resolved = await resolveServer(serverName, localGroupName, port);
      setResolving(false);

      if (!resolved.ok) {
        setError(
          resolved.error === 'not_found' ? t('auth.group_not_found') :
          resolved.error === 'unreachable' ? t('login2.server_unreachable') :
          t('auth.firebase_error')
        );
        setLoading(false);
        return;
      }

      setClientServerUrl(resolved.url);

      const success = await login({
        server_password: serverPassword,
        username: username.trim(),
        password,
        serverUrl: resolved.url,
      });

      if (success) {
        // Save login info if "remember me" is checked
        if (rememberMe) {
          localStorage.setItem('imagine-remember-me', 'true');
          localStorage.setItem('imagine-saved-login', JSON.stringify({
            serverName: serverName.trim(),
            serverPassword,
            username: username.trim(),
          }));
        } else {
          localStorage.removeItem('imagine-remember-me');
          localStorage.removeItem('imagine-saved-login');
        }

        // Determine mode: local admin → 'server', otherwise → 'client'
        const mode = resolved.local ? 'server' : 'client';
        onLoginComplete?.(mode);
      }
    } catch (err) {
      setError(err.message || t('auth.login') + ' failed');
    } finally {
      setLoading(false);
      setResolving(false);
    }
  }, [serverName, username, password, localGroupName, port, login, onLoginComplete, t]);

  // -----------------------------------------------------------------------
  // Register handler
  // -----------------------------------------------------------------------
  const handleRegister = useCallback(async (e) => {
    e.preventDefault();
    setError('');

    if (!serverName.trim()) { setError(t('login2.server_name_required')); return; }
    if (!serverPassword.trim()) { setError(t('validation.server_password_required')); return; }
    if (username.trim().length < 2) { setError(t('validation.username_min')); return; }
    if (password.length < 4) { setError(t('validation.password_min_4')); return; }

    setLoading(true);
    setResolving(true);

    try {
      const resolved = await resolveServer(serverName, localGroupName, port);
      setResolving(false);

      if (!resolved.ok) {
        setError(resolved.error === 'not_found' ? t('auth.group_not_found') : t('login2.server_unreachable'));
        setLoading(false);
        return;
      }

      setClientServerUrl(resolved.url);

      const success = await register({
        server_password: serverPassword,
        username: username.trim(),
        email: email.trim(),
        password,
        serverUrl: resolved.url,
      });

      if (success) {
        const mode = resolved.local ? 'server' : 'client';
        onLoginComplete?.(mode);
      }
    } catch (err) {
      setError(err.message || 'Registration failed');
    } finally {
      setLoading(false);
      setResolving(false);
    }
  }, [serverName, serverPassword, username, email, password, localGroupName, port, register, onLoginComplete, t]);

  // -----------------------------------------------------------------------
  // Existing server reset handlers
  // -----------------------------------------------------------------------

  /** Reset existing server (group only or full) then show create form. */
  const handleExistingReset = useCallback(async (resetAll) => {
    setError('');
    if (!oldServerPassword.trim()) {
      setError(t('validation.server_password_required'));
      return;
    }

    setLoading(true);
    try {
      const baseUrl = `http://localhost:${port}`;

      // Reset group (auth data) — uses server password, no JWT needed
      await resetGroup(baseUrl, oldServerPassword);

      if (resetAll) {
        // For full reset: init a temp admin, login, then reset database
        await initServer(baseUrl, {
          group_name: '__temp_reset__',
          server_password: oldServerPassword,
          admin_username: '__reset_admin__',
          admin_password: oldServerPassword,
        });
        const loggedIn = await login({
          server_password: oldServerPassword,
          username: '__reset_admin__',
          password: oldServerPassword,
          serverUrl: baseUrl,
        });
        if (loggedIn) {
          await resetDatabase(oldServerPassword);
          // Reset the temp group so we're back to uninitialized
          await resetGroup(baseUrl, oldServerPassword);
        }
      }

      setLocalGroupName(null);
      setExistingServerHandled(true);
      setOldServerPassword('');
    } catch (err) {
      setError(err.message || 'Reset failed');
    } finally {
      setLoading(false);
    }
  }, [oldServerPassword, port, login, t]);

  // -----------------------------------------------------------------------
  // Create server handler
  // -----------------------------------------------------------------------
  const handleCreateServer = useCallback(async (e) => {
    e.preventDefault();
    setError('');

    if (!newGroupName.trim()) { setError(t('validation.group_name_required')); return; }
    if (newServerPassword.length < 4) { setError(t('validation.password_min_4')); return; }
    if (adminUsername.trim().length < 2) { setError(t('validation.username_min')); return; }
    if (adminPassword.length < 4) { setError(t('validation.password_min_4')); return; }

    if (!isElectron) {
      setError(t('login2.create_requires_electron'));
      return;
    }

    setLoading(true);

    try {
      // Ensure local server is running
      if (!localServerReady) {
        await window.electron?.server?.start({ port });
        await new Promise(r => setTimeout(r, 2000));
      }

      const baseUrl = `http://localhost:${port}`;
      setClientServerUrl(baseUrl);

      // Set tier before init
      try {
        await window.electron?.pipeline?.setTier(selectedTier);
      } catch { /* ignore if not available */ }

      // Initialize server
      await initServer(baseUrl, {
        group_name: newGroupName.trim(),
        server_password: newServerPassword,
        admin_username: adminUsername.trim(),
        admin_password: adminPassword,
      });

      // Login with the new admin account
      const success = await login({
        server_password: newServerPassword,
        username: adminUsername.trim(),
        password: adminPassword,
        serverUrl: baseUrl,
      });

      if (success) {
        setLocalGroupName(newGroupName.trim());
        setLocalServerReady(true);
        onLoginComplete?.('server');
      }
    } catch (err) {
      setError(err.message || 'Server creation failed');
    } finally {
      setLoading(false);
    }
  }, [newGroupName, newServerPassword, adminUsername, adminPassword, selectedTier, port, localServerReady, login, onLoginComplete, t]);

  const displayError = error || authError;
  const togglePassword = () => setShowPassword(!showPassword);

  // Shared props for password fields
  const pwProps = { showPassword, onTogglePassword: togglePassword };

  // -----------------------------------------------------------------------
  // Login View
  // -----------------------------------------------------------------------
  const renderLogin = () => (
    <form onSubmit={handleLogin} className="space-y-4">
      <InputField
        icon={Server}
        label={t('login2.server_name')}
        value={serverName}
        onChange={setServerName}
        placeholder={t('login2.server_name_placeholder')}
        autoFocus
        disabled={loading}
      />
      <InputField
        label={t('auth.server_password')}
        type="password"
        value={serverPassword}
        onChange={setServerPassword}
        placeholder={t('auth.server_password_placeholder')}
        disabled={loading}
        {...pwProps}
      />
      {localGroupName && serverName.trim().toLowerCase() === localGroupName.toLowerCase() && (
        <div className="flex items-center gap-1.5 text-xs text-emerald-400 -mt-2">
          <CheckCircle size={12} />
          {t('login2.local_server')}
        </div>
      )}

      <div className="border-t border-zinc-700/40 pt-3" />

      <InputField
        label={t('auth.username')}
        value={username}
        onChange={setUsername}
        placeholder={t('auth.username_placeholder')}
        disabled={loading}
      />
      <InputField
        label={t('auth.password')}
        type="password"
        value={password}
        onChange={setPassword}
        placeholder="••••••"
        disabled={loading}
        {...pwProps}
      />

      <label className="flex items-center gap-2 cursor-pointer select-none">
        <input
          type="checkbox"
          checked={rememberMe}
          onChange={e => setRememberMe(e.target.checked)}
          className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 text-blue-500 focus:ring-blue-500/30 focus:ring-offset-0"
        />
        <span className="text-xs text-zinc-500">{t('login2.remember_me')}</span>
      </label>

      <SubmitButton loading={loading} resolving={resolving} resolvingText={t('login2.resolving')}>
        {t('auth.login')}
      </SubmitButton>

      <div className="flex items-center gap-3 text-xs text-zinc-500">
        <div className="flex-1 border-t border-zinc-700/50" />
        <span>{t('login2.or')}</span>
        <div className="flex-1 border-t border-zinc-700/50" />
      </div>

      <div className="flex gap-3">
        <button
          type="button"
          onClick={() => { setError(''); setView('register'); }}
          className="flex-1 py-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700/50 rounded-lg
            hover:border-zinc-600 transition-colors flex items-center justify-center gap-1.5"
        >
          <UserPlus size={13} />
          {t('login2.register')}
        </button>
        {isElectron && (
          <button
            type="button"
            onClick={() => { setError(''); setView('create'); }}
            className="flex-1 py-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700/50 rounded-lg
              hover:border-zinc-600 transition-colors flex items-center justify-center gap-1.5"
          >
            <Plus size={13} />
            {t('login2.create_server')}
          </button>
        )}
      </div>
    </form>
  );

  // -----------------------------------------------------------------------
  // Register View
  // -----------------------------------------------------------------------
  const renderRegister = () => (
    <form onSubmit={handleRegister} className="space-y-4">
      <button
        type="button"
        onClick={() => { setError(''); setView('login'); }}
        className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 mb-1"
      >
        <ArrowLeft size={13} /> {t('login2.back_to_login')}
      </button>

      <InputField
        icon={Server}
        label={t('login2.server_name')}
        value={serverName}
        onChange={setServerName}
        placeholder={t('login2.server_name_placeholder')}
        autoFocus={!serverName}
        disabled={loading}
      />
      <InputField
        label={t('auth.server_password')}
        type="password"
        value={serverPassword}
        onChange={setServerPassword}
        placeholder={t('auth.server_password_placeholder')}
        disabled={loading}
        {...pwProps}
      />
      <p className="text-[11px] text-zinc-500 -mt-2">{t('auth.server_password_hint')}</p>

      <div className="border-t border-zinc-700/40 pt-3">
        <p className="text-xs font-medium text-zinc-400 mb-3">{t('group.your_account')}</p>
        <div className="space-y-3">
          <InputField
            label={t('auth.username')}
            value={username}
            onChange={setUsername}
            placeholder={t('auth.username_placeholder')}
            disabled={loading}
          />
          <InputField
            label={t('auth.password')}
            type="password"
            value={password}
            onChange={setPassword}
            placeholder="••••••"
            disabled={loading}
            {...pwProps}
          />
          <InputField
            label={`${t('auth.email')} (${t('label.optional')})`}
            type="email"
            value={email}
            onChange={setEmail}
            placeholder="you@example.com"
            disabled={loading}
          />
        </div>
      </div>

      <SubmitButton loading={loading} resolving={resolving} resolvingText={t('login2.resolving')}>
        {t('auth.register')}
      </SubmitButton>
    </form>
  );

  // -----------------------------------------------------------------------
  // Create Server View
  // -----------------------------------------------------------------------
  const renderCreate = () => (
    <form onSubmit={handleCreateServer} className="space-y-4">
      <button
        type="button"
        onClick={() => { setError(''); setView('login'); }}
        className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 mb-1"
      >
        <ArrowLeft size={13} /> {t('login2.back_to_login')}
      </button>

      {/* Existing server warning */}
      {localGroupName && !existingServerHandled && (
        <div className="space-y-3">
          <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-lg">
            <div className="flex items-start gap-2 mb-3">
              <AlertTriangle size={15} className="text-amber-400 mt-0.5 shrink-0" />
              <p className="text-xs text-amber-300">
                {t('login2.existing_server_warning', { name: localGroupName })}
              </p>
            </div>

            <InputField
              label={t('login2.existing_server_old_password')}
              type="password"
              value={oldServerPassword}
              onChange={setOldServerPassword}
              placeholder="••••••"
              disabled={loading}
              {...pwProps}
            />
          </div>

          <div className="space-y-2">
            <button
              type="button"
              disabled={loading || !oldServerPassword.trim()}
              onClick={() => handleExistingReset(true)}
              className="w-full p-2.5 text-left bg-red-500/10 border border-red-500/30 rounded-lg
                hover:bg-red-500/15 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <div className="flex items-center gap-2">
                <Trash2 size={14} className="text-red-400 shrink-0" />
                <div>
                  <p className="text-xs font-medium text-red-300">{t('login2.existing_reset_all')}</p>
                  <p className="text-[11px] text-zinc-500 mt-0.5">{t('login2.existing_reset_all_desc')}</p>
                </div>
              </div>
              {loading && <Loader2 size={14} className="text-red-400 animate-spin ml-auto" />}
            </button>

            <button
              type="button"
              disabled={loading || !oldServerPassword.trim()}
              onClick={() => handleExistingReset(false)}
              className="w-full p-2.5 text-left bg-blue-500/10 border border-blue-500/30 rounded-lg
                hover:bg-blue-500/15 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <div className="flex items-center gap-2">
                <RefreshCw size={14} className="text-blue-400 shrink-0" />
                <div>
                  <p className="text-xs font-medium text-blue-300">{t('login2.existing_reset_group')}</p>
                  <p className="text-[11px] text-zinc-500 mt-0.5">{t('login2.existing_reset_group_desc')}</p>
                </div>
              </div>
            </button>
          </div>
        </div>
      )}

      {/* Server config + form — hidden when existing server needs handling */}
      {(!localGroupName || existingServerHandled) && <>
        <div className="space-y-3">
          <p className="text-xs font-medium text-zinc-400">{t('login2.server_config')}</p>
          <InputField
            icon={Server}
            label={t('group.name')}
            value={newGroupName}
            onChange={setNewGroupName}
            placeholder={t('group.name_placeholder')}
            autoFocus
            disabled={loading}
          />
          <InputField
            label={t('group.server_password')}
            type="password"
            value={newServerPassword}
            onChange={setNewServerPassword}
            placeholder={t('group.server_password_placeholder')}
            disabled={loading}
            {...pwProps}
          />
          <p className="text-[11px] text-zinc-500 -mt-2">{t('group.server_password_hint')}</p>
        </div>

        {/* Admin account */}
        <div className="border-t border-zinc-700/40 pt-3 space-y-3">
          <p className="text-xs font-medium text-zinc-400">{t('group.admin_account')}</p>
          <InputField
            label={t('group.admin_username')}
            value={adminUsername}
            onChange={setAdminUsername}
            placeholder={t('group.admin_username_placeholder')}
            disabled={loading}
          />
          <InputField
            label={t('group.admin_password')}
            type="password"
            value={adminPassword}
            onChange={setAdminPassword}
            placeholder="••••••"
            disabled={loading}
            {...pwProps}
          />
        </div>

        {/* Tier selection */}
        <div className="border-t border-zinc-700/40 pt-3">
          <p className="text-xs font-medium text-zinc-400 mb-2">{t('setup.select_tier')}</p>
          <div className="grid grid-cols-3 gap-2">
            {TIERS.map(tier => {
              const Icon = tier.icon;
              const selected = selectedTier === tier.id;
              return (
                <button
                  key={tier.id}
                  type="button"
                  onClick={() => setSelectedTier(tier.id)}
                  className={`p-2.5 rounded-lg border text-center transition-all ${
                    selected
                      ? `border-${tier.color}-500/60 bg-${tier.color}-500/10`
                      : 'border-zinc-700/50 hover:border-zinc-600'
                  }`}
                >
                  <Icon size={18} className={`mx-auto mb-1 ${selected ? `text-${tier.color}-400` : 'text-zinc-500'}`} />
                  <div className={`text-xs font-medium ${selected ? 'text-zinc-200' : 'text-zinc-500'}`}>
                    {t(`setup.tier_${tier.id}_title`)}
                  </div>
                  <div className="text-[10px] text-zinc-600 mt-0.5">
                    {t(`setup.tier_${tier.id}_vram`)}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        <SubmitButton loading={loading}>
          <Plus size={15} />
          {t('login2.create_and_start')}
        </SubmitButton>
      </>}
    </form>
  );

  // -----------------------------------------------------------------------
  // Main render
  // -----------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-zinc-900 flex items-center justify-center p-4">
      <div className="w-full max-w-sm">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-zinc-100 tracking-tight">Imagine</h1>
          <p className="text-xs text-zinc-500 mt-1">{t('auth.subtitle')}</p>
        </div>

        {/* Card */}
        <div className="bg-zinc-800/40 border border-zinc-700/40 rounded-xl p-5 shadow-xl">
          {/* View title */}
          <h2 className="text-sm font-semibold text-zinc-200 mb-4">
            {view === 'login' && t('auth.login')}
            {view === 'register' && t('auth.register')}
            {view === 'create' && t('login2.create_server')}
          </h2>

          {/* Error display */}
          {displayError && (
            <div className="mb-4 p-2.5 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-2">
              <AlertCircle size={14} className="text-red-400 mt-0.5 shrink-0" />
              <p className="text-xs text-red-300">{displayError}</p>
            </div>
          )}

          {/* Local server status */}
          {localServerStarting && (
            <div className="mb-4 p-2.5 bg-blue-500/10 border border-blue-500/30 rounded-lg flex items-center gap-2">
              <Loader2 size={14} className="text-blue-400 animate-spin" />
              <p className="text-xs text-blue-300">{t('login2.detecting_local')}</p>
            </div>
          )}
          {!localServerStarting && isElectron && view === 'login' && (
            localGroupName ? (
              <div className="mb-4 p-2.5 bg-emerald-500/10 border border-emerald-500/30 rounded-lg flex items-center gap-2">
                <Server size={14} className="text-emerald-400 shrink-0" />
                <p className="text-xs text-emerald-300 flex-1">
                  {t('login2.local_server_available', { name: localGroupName })}
                </p>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => { setError(''); setView('create'); }}
                className="mb-4 w-full p-2.5 bg-blue-500/10 border border-blue-500/30 rounded-lg
                  flex items-center gap-2 hover:bg-blue-500/15 transition-colors"
              >
                <Plus size={14} className="text-blue-400 shrink-0" />
                <p className="text-xs text-blue-300">{t('login2.create_server_here')}</p>
              </button>
            )
          )}

          {/* Views */}
          {view === 'login' && renderLogin()}
          {view === 'register' && renderRegister()}
          {view === 'create' && renderCreate()}
        </div>

        {/* Language switcher */}
        <div className="flex items-center justify-center gap-2 mt-4">
          <Languages size={13} className="text-zinc-600" />
          {availableLocales.map(loc => (
            <button
              key={loc}
              onClick={() => setLocale(loc)}
              className={`text-xs px-2 py-0.5 rounded transition-colors ${
                locale === loc ? 'text-zinc-200 bg-zinc-700/50' : 'text-zinc-600 hover:text-zinc-400'
              }`}
            >
              {loc === 'en-US' ? 'EN' : 'KO'}
            </button>
          ))}
        </div>

        {/* Version */}
        <p className="text-center text-[10px] text-zinc-700 mt-2">v0.6.4</p>
      </div>
    </div>
  );
}
