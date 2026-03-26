/**
 * LoginPageV2 — Two-layer auth login.
 *
 * Stage 1 (auth): Firebase Auth — email/password personal account
 * Stage 2 (server_select): Server name + server password → connect
 * Stage 2 (server_init): Create new server (Electron only)
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLocale } from '../i18n';
import { isElectron, setServerUrl as setClientServerUrl } from '../api/client';
import { signUp, signIn, signInWithGoogle, getIdToken } from '../api/firebaseAuth';
import { getServerInfo, initServer } from '../api/auth';
import { registerGroup } from '../api/firebase';
import { getServerHistory, addServerToHistory } from '../utils/serverHistory';
import {
  LogIn, UserPlus, Eye, EyeOff, Loader2, ArrowLeft,
  Zap, Star, Rocket, Languages, AlertCircle, Server,
  LogOut, Key, Mail, Shield, X,
} from 'lucide-react';

const TIERS = [
  { id: 'standard', icon: Zap, color: 'emerald' },
  { id: 'pro', icon: Star, color: 'blue' },
  { id: 'ultra', icon: Rocket, color: 'purple' },
];

// ---------------------------------------------------------------------------
// Shared UI
// ---------------------------------------------------------------------------

function InputField({ icon: Icon, label, type = 'text', value, onChange, placeholder, autoFocus, disabled, showPassword, onTogglePassword, inputRef }) {
  const isPassword = type === 'password';
  const actualType = isPassword && showPassword ? 'text' : type;
  return (
    <div>
      {label && <label className="block text-xs font-medium text-zinc-400 mb-1.5">{label}</label>}
      <div className="relative">
        {Icon && <Icon size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-zinc-500" />}
        <input
          ref={inputRef}
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

function SubmitButton({ children, loading, disabled }) {
  return (
    <button
      type="submit"
      disabled={disabled || loading}
      className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500
        text-white text-sm font-medium rounded-lg transition-colors
        flex items-center justify-center gap-2"
    >
      {loading ? <Loader2 size={16} className="animate-spin" /> : null}
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function LoginPageV2({ onLoginComplete, serverPort }) {
  const {
    firebaseUser, isFirebaseAuthenticated,
    connectToServer, fullLogout,
    error: authError, clearError: clearAuthError,
  } = useAuth();
  const { t, locale, setLocale, availableLocales } = useLocale();

  // --- Stage: 'auth' | 'server_select' | 'server_init' ---
  const [stage, setStage] = useState('auth');

  // --- Auth sub-view: 'login' | 'register' ---
  const [authView, setAuthView] = useState('login');

  // --- Local server detection ---
  const [localGroupName, setLocalGroupName] = useState(null);
  const [localServerReady, setLocalServerReady] = useState(false);
  const [localServerStarting, setLocalServerStarting] = useState(false);

  // --- Form fields (Firebase auth) ---
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // --- Server connect fields ---
  const [serverName, setServerName] = useState(() => {
    const history = getServerHistory();
    return history.length > 0 ? (history[0].name || '') : '';
  });
  const [serverPassword, setServerPassword] = useState('');
  const [isServerAdmin, setIsServerAdmin] = useState(false);
  const [showServerPassword, setShowServerPassword] = useState(false);

  // --- Create server fields ---
  const [newGroupName, setNewGroupName] = useState('');
  const [newServerPassword, setNewServerPassword] = useState('');
  const [newServerPasswordConfirm, setNewServerPasswordConfirm] = useState('');
  const [selectedTier, setSelectedTier] = useState('pro');
  const [showNewPassword, setShowNewPassword] = useState(false);

  // --- Status ---
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState([]);
  const _errIdRef = useRef(0);
  const setError = useCallback((msg) => {
    if (!msg) { setErrors([]); return; }
    setErrors(prev => [...prev, { id: ++_errIdRef.current, msg }]);
  }, []);
  const dismissError = useCallback((id) => {
    setErrors(prev => prev.filter(e => e.id !== id));
  }, []);

  const port = serverPort || 8000;
  const serverPwRef = useRef(null);

  // --- Server history (localStorage) ---
  const [serverHistory, setServerHistory] = useState(() => getServerHistory());

  const addToHistory = useCallback((name, url) => {
    addServerToHistory({ url: url || '', name });
    setServerHistory(getServerHistory());
  }, []);

  // ── When Firebase user is available, switch to server_select stage ──
  useEffect(() => {
    if (isFirebaseAuthenticated) {
      setStage('server_select');
    } else {
      setStage('auth');
    }
  }, [isFirebaseAuthenticated]);

  // ── Local server detection (Electron + isServerAdmin) ──
  useEffect(() => {
    if (!isElectron || !isServerAdmin) {
      setLocalServerReady(false);
      setLocalGroupName(null);
      return;
    }

    const detectLocal = async () => {
      try {
        const status = await window.electron?.server?.getStatus();
        if (status?.running) {
          setLocalServerReady(true);
          const info = await getServerInfo(`http://localhost:${port}`);
          if (info.ok && info.group_name) {
            setLocalGroupName(info.group_name);
          }
          return;
        }

        setLocalServerStarting(true);
        await window.electron?.server?.start({ port });
        await new Promise(r => setTimeout(r, 2000));

        const info = await getServerInfo(`http://localhost:${port}`);
        if (info.ok) {
          setLocalServerReady(true);
          if (info.initialized && info.group_name) {
            setLocalGroupName(info.group_name);
          }
        } else {
          try { await window.electron?.server?.stop(); } catch { /* ignore */ }
        }
      } catch (e) {
        console.error('Local server detection failed:', e);
      } finally {
        setLocalServerStarting(false);
      }
    };

    detectLocal();
  }, [isServerAdmin, port]);

  // -----------------------------------------------------------------------
  // Firebase Auth handlers
  // -----------------------------------------------------------------------
  const handleGoogleLogin = useCallback(async () => {
    setError('');
    setLoading(true);
    try {
      await signInWithGoogle();
    } catch (err) {
      if (err.code !== 'auth/popup-closed-by-user') {
        setError(err.message || t('auth.login_failed'));
      }
    } finally {
      setLoading(false);
    }
  }, [t]);

  const handleFirebaseLogin = useCallback(async (e) => {
    e.preventDefault();
    setError('');
    if (!email.trim()) { setError(t('validation.email_required')); return; }
    if (!password) { setError(t('validation.field_required')); return; }

    setLoading(true);
    try {
      await signIn(email.trim(), password);
    } catch (err) {
      const code = err.code;
      if (code === 'auth/user-not-found' || code === 'auth/wrong-password' || code === 'auth/invalid-credential') {
        setError(t('auth.invalid_credentials'));
      } else if (code === 'auth/too-many-requests') {
        setError(t('auth.too_many_attempts'));
      } else {
        setError(err.message || t('auth.login_failed'));
      }
    } finally {
      setLoading(false);
    }
  }, [email, password, t]);

  const handleFirebaseRegister = useCallback(async (e) => {
    e.preventDefault();
    setError('');
    if (!email.trim()) { setError(t('validation.email_required')); return; }
    if (password.length < 6) { setError(t('validation.password_min_6')); return; }

    setLoading(true);
    try {
      await signUp(email.trim(), password, displayName.trim() || undefined);
    } catch (err) {
      const code = err.code;
      if (code === 'auth/email-already-in-use') {
        setError(t('auth.email_in_use'));
      } else if (code === 'auth/weak-password') {
        setError(t('validation.password_min_6'));
      } else {
        setError(err.message || t('auth.register_failed'));
      }
    } finally {
      setLoading(false);
    }
  }, [email, password, displayName, t]);

  // -----------------------------------------------------------------------
  // Server connect handler
  // -----------------------------------------------------------------------
  const handleConnect = useCallback(async (e) => {
    e.preventDefault();
    setError('');
    const name = serverName.trim();
    if (!name) { setError(t('login2.server_name_required')); return; }
    if (!serverPassword) { setError(t('login2.server_password_required')); return; }

    setLoading(true);
    try {
      let directUrl = null;

      if (isServerAdmin && isElectron) {
        // Server admin (Electron): ensure server is running, then connect to localhost
        directUrl = `http://localhost:${port}`;

        if (!localServerReady) {
          // Start server if not running yet
          setLocalServerStarting(true);
          try {
            await window.electron?.server?.start({ port });
            // Wait for server to be ready (poll health endpoint)
            let ready = false;
            for (let i = 0; i < 15; i++) {
              await new Promise(r => setTimeout(r, 1000));
              try {
                const resp = await fetch(`${directUrl}/api/v1/health`, { signal: AbortSignal.timeout(2000) });
                if (resp.ok) { ready = true; break; }
              } catch { /* server not ready yet */ }
            }
            if (!ready) {
              setError('Server failed to start within 15 seconds');
              setLoading(false);
              return;
            }
            setLocalServerReady(true);
          } finally {
            setLocalServerStarting(false);
          }
        }

        setClientServerUrl(directUrl);

        // Check if server is initialized
        const info = await getServerInfo(directUrl);
        if (info.ok && !info.initialized) {
          // Not initialized — auto-initialize with entered group name + password
          const adminName = firebaseUser?.displayName || firebaseUser?.email?.split('@')[0] || 'admin';
          await initServer(directUrl, {
            group_name: name,
            server_password: serverPassword,
            admin_username: adminName,
            admin_password: serverPassword,
            firebase_uid: firebaseUser?.uid || null,
            firebase_email: firebaseUser?.email || null,
          });

          // Register to Firestore for external discovery
          try {
            const localIp = await window.electron?.network?.getLocalIp?.();
            await registerGroup(name, { lan_ip: localIp || '', port });
          } catch (regErr) {
            if (regErr.message === 'GROUP_NAME_TAKEN') {
              setError(t('validation.group_name_taken'));
              setLoading(false);
              return;
            }
          }
        }
      }

      const success = await connectToServer(name, serverPassword, directUrl);
      if (success) {
        const { getServerUrl } = await import('../api/client');
        addToHistory(name, getServerUrl());
        const mode = isServerAdmin ? 'server' : 'client';
        onLoginComplete?.(mode);
      }
    } catch (err) {
      setError(err.message || 'Connection failed');
    } finally {
      setLoading(false);
    }
  }, [serverName, serverPassword, isServerAdmin, localServerReady, port, firebaseUser, connectToServer, addToHistory, onLoginComplete, t]);

  // -----------------------------------------------------------------------
  // Create server handler (Electron only)
  // -----------------------------------------------------------------------
  const handleCreateServer = useCallback(async (e) => {
    e.preventDefault();
    setError('');
    const name = newGroupName.trim();
    if (!name) { setError(t('validation.group_name_required')); return; }
    if (!newServerPassword) { setError(t('login2.server_password_required')); return; }
    if (newServerPassword !== newServerPasswordConfirm) { setError(t('login2.password_mismatch')); return; }

    setLoading(true);
    try {
      if (!localServerReady) {
        await window.electron?.server?.start({ port });
        await new Promise(r => setTimeout(r, 2000));
        setLocalServerReady(true);
      }

      const baseUrl = `http://localhost:${port}`;
      setClientServerUrl(baseUrl);

      // Set tier
      try {
        await window.electron?.pipeline?.setTier(selectedTier);
      } catch { /* ignore */ }

      // Initialize server — Firebase user becomes admin automatically
      const adminName = firebaseUser?.displayName || firebaseUser?.email?.split('@')[0] || 'admin';
      await initServer(baseUrl, {
        group_name: name,
        server_password: newServerPassword,
        admin_username: adminName,
        admin_password: newServerPassword,
        firebase_uid: firebaseUser?.uid || null,
        firebase_email: firebaseUser?.email || null,
      });

      // Register to Firestore for discovery (reject duplicate names)
      try {
        const localIp = await window.electron?.network?.getLocalIp?.();
        await registerGroup(name, {
          lan_ip: localIp || '',
          port,
        });
      } catch (regErr) {
        if (regErr.message === 'GROUP_NAME_TAKEN') {
          setError(t('validation.group_name_taken'));
          setLoading(false);
          return;
        }
        // Other registration errors are best-effort (network issues etc.)
      }

      // Connect using 2-layer auth
      const success = await connectToServer(name, newServerPassword, baseUrl);
      if (success) {
        addToHistory(name, baseUrl);
        setLocalGroupName(name);
        onLoginComplete?.('server');
      }
    } catch (err) {
      setError(err.message || 'Server creation failed');
    } finally {
      setLoading(false);
    }
  }, [newGroupName, newServerPassword, newServerPasswordConfirm, firebaseUser, selectedTier, port, localServerReady, connectToServer, addToHistory, onLoginComplete, t]);

  const handleFullLogout = useCallback(async () => {
    await fullLogout();
    setStage('auth');
    setEmail('');
    setPassword('');
  }, [fullLogout]);

  const displayErrors = [...errors];
  if (authError && !errors.some(e => e.msg === authError)) {
    displayErrors.push({ id: 'auth', msg: authError });
  }
  const togglePassword = () => setShowPassword(!showPassword);
  const pwProps = { showPassword, onTogglePassword: togglePassword };

  // -----------------------------------------------------------------------
  // Stage 1: Firebase Auth
  // -----------------------------------------------------------------------
  const renderAuthStage = () => (
    <>
      {authView === 'login' ? (
        <div className="space-y-4">
          {/* Google login */}
          <button
            type="button"
            onClick={handleGoogleLogin}
            disabled={loading}
            className="w-full py-2.5 bg-white hover:bg-gray-100 disabled:bg-zinc-700
              text-gray-800 text-sm font-medium rounded-lg transition-colors
              flex items-center justify-center gap-2.5 border border-gray-300"
          >
            <svg width="16" height="16" viewBox="0 0 48 48"><path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/><path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/><path fill="#FBBC05" d="M10.53 28.59a14.5 14.5 0 0 1 0-9.18l-7.98-6.19a24.01 24.01 0 0 0 0 21.56l7.98-6.19z"/><path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/></svg>
            {t('auth.google_login')}
          </button>

          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <div className="flex-1 border-t border-zinc-700/50" />
            <span>{t('login2.or')}</span>
            <div className="flex-1 border-t border-zinc-700/50" />
          </div>

          {/* Email login */}
          <form onSubmit={handleFirebaseLogin} className="space-y-3">
            <InputField
              icon={Mail}
              label={t('auth.email')}
              type="email"
              value={email}
              onChange={setEmail}
              placeholder="you@example.com"
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

            <SubmitButton loading={loading}>
              <LogIn size={15} />
              {t('auth.login')}
            </SubmitButton>
          </form>

          <button
            type="button"
            onClick={() => { setError(''); setAuthView('register'); }}
            className="w-full py-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700/50 rounded-lg
              hover:border-zinc-600 transition-colors flex items-center justify-center gap-1.5"
          >
            <UserPlus size={13} />
            {t('auth.create_account')}
          </button>
        </div>
      ) : (
        <form onSubmit={handleFirebaseRegister} className="space-y-4">
          <button
            type="button"
            onClick={() => { setError(''); setAuthView('login'); }}
            className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 mb-1"
          >
            <ArrowLeft size={13} /> {t('login2.back_to_login')}
          </button>

          <InputField
            icon={Mail}
            label={t('auth.email')}
            type="email"
            value={email}
            onChange={setEmail}
            placeholder="you@example.com"
            autoFocus
            disabled={loading}
          />
          <InputField
            label={t('auth.display_name') + ` (${t('label.optional')})`}
            value={displayName}
            onChange={setDisplayName}
            placeholder={t('auth.display_name_placeholder')}
            disabled={loading}
          />
          <InputField
            label={t('auth.password')}
            type="password"
            value={password}
            onChange={setPassword}
            placeholder={t('auth.password_min_6_hint')}
            disabled={loading}
            {...pwProps}
          />

          <SubmitButton loading={loading}>
            <UserPlus size={15} />
            {t('auth.create_account')}
          </SubmitButton>
        </form>
      )}
    </>
  );

  // -----------------------------------------------------------------------
  // Stage 2: Server Select
  // -----------------------------------------------------------------------
  const renderServerSelect = () => (
    <div className="space-y-4">
      {/* User info bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-7 h-7 rounded-full bg-blue-600/30 flex items-center justify-center text-xs text-blue-300 font-medium shrink-0">
            {(firebaseUser?.displayName || firebaseUser?.email || '?')[0].toUpperCase()}
          </div>
          <div className="min-w-0">
            <p className="text-xs text-zinc-300 truncate">{firebaseUser?.displayName || firebaseUser?.email}</p>
            {firebaseUser?.displayName && (
              <p className="text-[10px] text-zinc-500 truncate">{firebaseUser?.email}</p>
            )}
          </div>
        </div>
        <button
          onClick={handleFullLogout}
          className="text-zinc-500 hover:text-zinc-300 transition-colors p-1"
          title={t('auth.logout')}
        >
          <LogOut size={14} />
        </button>
      </div>

      <div className="border-t border-zinc-700/40" />

      {/* Server connect form */}
      <form onSubmit={handleConnect} className="space-y-3">
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
          icon={Key}
          label={t('login2.server_password')}
          type="password"
          value={serverPassword}
          onChange={setServerPassword}
          placeholder="••••••"
          autoFocus={!!serverName}
          disabled={loading}
          showPassword={showServerPassword}
          onTogglePassword={() => setShowServerPassword(!showServerPassword)}
          inputRef={serverPwRef}
        />

        {/* Server admin checkbox (Electron only) */}
        {isElectron && (
          <label className="flex items-center gap-2 cursor-pointer group">
            <input
              type="checkbox"
              checked={isServerAdmin}
              onChange={e => setIsServerAdmin(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 text-blue-500 focus:ring-blue-500/30"
            />
            <div className="flex items-center gap-1.5">
              <Shield size={13} className={isServerAdmin ? 'text-blue-400' : 'text-zinc-500'} />
              <span className={`text-xs ${isServerAdmin ? 'text-zinc-200' : 'text-zinc-500'}`}>
                {t('login2.server_admin')}
              </span>
            </div>
          </label>
        )}

        {/* Local server status */}
        {isServerAdmin && localServerStarting && (
          <div className="p-2 bg-blue-500/10 border border-blue-500/30 rounded-lg flex items-center gap-2">
            <Loader2 size={14} className="text-blue-400 animate-spin" />
            <p className="text-xs text-blue-300">{t('login2.starting_local_server')}</p>
          </div>
        )}

        {isServerAdmin && localServerReady && localGroupName && (
          <div className="p-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg flex items-center gap-2">
            <Server size={14} className="text-emerald-400" />
            <p className="text-xs text-emerald-300">{localGroupName}</p>
          </div>
        )}

        <SubmitButton loading={loading}>
          <LogIn size={15} />
          {t('login2.connect')}
        </SubmitButton>
      </form>

      {/* Create new server (Electron only) */}
      {isElectron && (
        <>
          <div className="flex items-center gap-3 text-xs text-zinc-600">
            <div className="flex-1 border-t border-zinc-700/40" />
            <span>{t('login2.or')}</span>
            <div className="flex-1 border-t border-zinc-700/40" />
          </div>
          <button
            type="button"
            onClick={() => { setError(''); setStage('server_init'); }}
            className="w-full py-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700/50 rounded-lg
              hover:border-zinc-600 transition-colors flex items-center justify-center gap-1.5"
          >
            <Server size={13} />
            {t('login2.create_new_server')}
          </button>
        </>
      )}
    </div>
  );

  // -----------------------------------------------------------------------
  // Stage 2b: Server Init (Electron only)
  // -----------------------------------------------------------------------
  const renderServerInit = () => (
    <form onSubmit={handleCreateServer} className="space-y-4">
      <button
        type="button"
        onClick={() => { setError(''); setStage('server_select'); }}
        className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 mb-1"
      >
        <ArrowLeft size={13} /> {t('login2.back_to_server_select')}
      </button>

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
        icon={Key}
        label={t('login2.server_password')}
        type="password"
        value={newServerPassword}
        onChange={setNewServerPassword}
        placeholder={t('login2.set_server_password')}
        disabled={loading}
        showPassword={showNewPassword}
        onTogglePassword={() => setShowNewPassword(!showNewPassword)}
      />
      <InputField
        icon={Key}
        label={t('login2.confirm_password')}
        type="password"
        value={newServerPasswordConfirm}
        onChange={setNewServerPasswordConfirm}
        placeholder={t('login2.confirm_password_placeholder')}
        disabled={loading}
        showPassword={showNewPassword}
        onTogglePassword={() => setShowNewPassword(!showNewPassword)}
      />

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
        <Server size={15} />
        {t('login2.create_and_start')}
      </SubmitButton>
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
          <p className="text-xs text-zinc-500 mt-1">
            {stage === 'auth' ? t('auth.subtitle') :
             stage === 'server_init' ? t('login2.create_new_server') :
             t('login2.select_server')}
          </p>
        </div>

        {/* Card */}
        <div className="bg-zinc-800/40 border border-zinc-700/40 rounded-xl p-5 shadow-xl">
          {/* View title */}
          <h2 className="text-sm font-semibold text-zinc-200 mb-4">
            {stage === 'auth' && (
              authView === 'login' ? t('auth.login') : t('auth.create_account')
            )}
            {stage === 'server_select' && t('login2.select_server')}
            {stage === 'server_init' && t('login2.create_new_server')}
          </h2>

          {/* Error display — each error can be dismissed independently */}
          {displayErrors.length > 0 && (
            <div className="mb-4 space-y-2">
              {displayErrors.map(e => (
                <div key={e.id} className="p-2.5 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-2">
                  <AlertCircle size={14} className="text-red-400 mt-0.5 shrink-0" />
                  <p className="text-xs text-red-300 flex-1">{e.msg}</p>
                  <button
                    type="button"
                    onClick={() => e.id === 'auth' ? clearAuthError() : dismissError(e.id)}
                    className="text-red-400/60 hover:text-red-300 shrink-0"
                  >
                    <X size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Stage content */}
          {stage === 'auth' && renderAuthStage()}
          {stage === 'server_select' && renderServerSelect()}
          {stage === 'server_init' && renderServerInit()}
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
