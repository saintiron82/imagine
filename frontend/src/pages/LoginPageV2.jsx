/**
 * LoginPageV2 — Two-stage Firebase Auth login.
 *
 * Stage 1: Firebase Auth (email/password) — personal account
 * Stage 2: Group selection / join / create
 *
 * Electron local mode uses legacy auth as fallback.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useLocale } from '../i18n';
import { isElectron, setServerUrl as setClientServerUrl } from '../api/client';
import { lookupGroup } from '../api/firebase';
import { signUp, signIn, signOut, getIdToken } from '../api/firebaseAuth';
import { getServerInfo, firebaseInitServer } from '../api/auth';
import {
  LogIn, UserPlus, Plus, Eye, EyeOff, Loader2, ArrowLeft,
  Zap, Star, Rocket, Languages, AlertCircle, CheckCircle, Server,
  Users, LogOut, Key, Mail,
} from 'lucide-react';

const TIERS = [
  { id: 'standard', icon: Zap, color: 'emerald' },
  { id: 'pro', icon: Star, color: 'blue' },
  { id: 'ultra', icon: Rocket, color: 'purple' },
];

// ---------------------------------------------------------------------------
// Shared UI
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
// Helpers
// ---------------------------------------------------------------------------

async function resolveServer(serverName, localGroupName, localPort) {
  const trimmed = serverName.trim();
  if (!trimmed) return { ok: false, error: 'empty' };

  if (localGroupName && trimmed.toLowerCase() === localGroupName.toLowerCase()) {
    return { ok: true, url: `http://localhost:${localPort || 8000}`, local: true };
  }

  try {
    const result = await lookupGroup(trimmed);
    if (!result) return { ok: false, error: 'not_found' };

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

    if (candidates.length > 0) return { ok: true, url: candidates[0], local: false };
    return { ok: false, error: 'unreachable' };
  } catch {
    return { ok: false, error: 'firebase_error' };
  }
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function LoginPageV2({ onLoginComplete, serverPort }) {
  const {
    firebaseUser, isFirebaseAuthenticated,
    connectToGroup, joinGroupWithCode, login,
    fullLogout, disconnectGroup,
    error: authError,
  } = useAuth();
  const { t, locale, setLocale, availableLocales } = useLocale();

  // --- Stage: 'auth' (Firebase login) | 'groups' (group selection) ---
  const [stage, setStage] = useState('auth');

  // --- Auth sub-view: 'login' | 'register' ---
  const [authView, setAuthView] = useState('login');

  // --- Group sub-view: 'list' | 'join' | 'create' ---
  const [groupView, setGroupView] = useState('list');

  // --- Local server detection ---
  const [localGroupName, setLocalGroupName] = useState(null);
  const [localServerReady, setLocalServerReady] = useState(false);
  const [localServerStarting, setLocalServerStarting] = useState(false);

  // --- Form fields (Firebase auth) ---
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  // --- Group join fields ---
  const [inviteCode, setInviteCode] = useState('');
  const [joinGroupName, setJoinGroupName] = useState('');

  // --- Create server fields ---
  const [newGroupName, setNewGroupName] = useState('');
  const [selectedTier, setSelectedTier] = useState('pro');

  // --- My groups (from Firebase RTDB) ---
  const [myGroups, setMyGroups] = useState([]);
  const [groupsLoading, setGroupsLoading] = useState(false);

  // --- Status ---
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [connectingGroup, setConnectingGroup] = useState(null);

  const port = serverPort || 8000;

  // ── When Firebase user is available, switch to groups stage ──
  useEffect(() => {
    if (isFirebaseAuthenticated) {
      setStage('groups');
      loadMyGroups();
    } else {
      setStage('auth');
    }
  }, [isFirebaseAuthenticated]);

  // ── Local server detection (Electron) ──
  useEffect(() => {
    if (!isElectron) return;

    const detectLocal = async () => {
      try {
        const status = await window.electron?.server?.getStatus();
        if (status?.running) {
          setLocalServerReady(true);
          setClientServerUrl(`http://localhost:${port}`);
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
        if (info.ok && info.initialized && info.group_name) {
          setLocalGroupName(info.group_name);
          setLocalServerReady(true);
          setClientServerUrl(`http://localhost:${port}`);
        } else if (info.ok && !info.initialized) {
          setLocalServerReady(true);
          setClientServerUrl(`http://localhost:${port}`);
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
  }, [port]);

  // ── Load user's groups from Firebase RTDB ──
  const loadMyGroups = useCallback(async () => {
    if (!firebaseUser) return;
    setGroupsLoading(true);
    try {
      const uid = firebaseUser.uid;
      const resp = await fetch(
        `https://imagine-b1e9c-default-rtdb.firebaseio.com/users/${uid}/groups.json`,
        { signal: AbortSignal.timeout(5000) }
      );
      if (resp.ok) {
        const data = await resp.json();
        if (data) {
          const groups = Object.entries(data).map(([key, val]) => ({
            key,
            group_name: val.group_name,
            role: val.role || 'user',
            joined_at: val.joined_at,
          }));
          setMyGroups(groups);
        } else {
          setMyGroups([]);
        }
      }
    } catch (e) {
      console.error('Failed to load groups:', e);
    } finally {
      setGroupsLoading(false);
    }
  }, [firebaseUser]);

  // ── Save group to user's Firebase RTDB ──
  const saveGroupToFirebase = useCallback(async (groupName, role) => {
    if (!firebaseUser) return;
    const uid = firebaseUser.uid;
    const key = groupName.trim().toLowerCase().replace(/\s+/g, '_');
    try {
      await fetch(
        `https://imagine-b1e9c-default-rtdb.firebaseio.com/users/${uid}/groups/${key}.json`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            group_name: groupName,
            role,
            joined_at: new Date().toISOString(),
          }),
          signal: AbortSignal.timeout(5000),
        }
      );
    } catch (e) {
      console.error('Failed to save group to Firebase:', e);
    }
  }, [firebaseUser]);

  // -----------------------------------------------------------------------
  // Firebase Auth handlers
  // -----------------------------------------------------------------------
  const handleFirebaseLogin = useCallback(async (e) => {
    e.preventDefault();
    setError('');
    if (!email.trim()) { setError(t('validation.email_required')); return; }
    if (!password) { setError(t('validation.field_required')); return; }

    setLoading(true);
    try {
      await signIn(email.trim(), password);
      // onAuthStateChanged will trigger stage switch
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
      // onAuthStateChanged will trigger stage switch
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
  // Group handlers
  // -----------------------------------------------------------------------
  const handleConnectGroup = useCallback(async (groupName) => {
    setError('');
    setConnectingGroup(groupName);

    try {
      const resolved = await resolveServer(groupName, localGroupName, port);
      if (!resolved.ok) {
        setError(
          resolved.error === 'not_found' ? t('auth.group_not_found') :
          resolved.error === 'unreachable' ? t('login2.server_unreachable') :
          t('auth.firebase_error')
        );
        return;
      }

      setClientServerUrl(resolved.url);
      const success = await connectToGroup(resolved.url, groupName);

      if (success) {
        const mode = resolved.local ? 'server' : 'client';
        onLoginComplete?.(mode);
      }
    } catch (err) {
      setError(err.message || 'Connection failed');
    } finally {
      setConnectingGroup(null);
    }
  }, [localGroupName, port, connectToGroup, onLoginComplete, t]);

  const handleJoinGroup = useCallback(async (e) => {
    e.preventDefault();
    setError('');
    if (!joinGroupName.trim()) { setError(t('login2.server_name_required')); return; }
    if (!inviteCode.trim()) { setError(t('validation.invite_code_required')); return; }

    setLoading(true);
    try {
      const resolved = await resolveServer(joinGroupName, localGroupName, port);
      if (!resolved.ok) {
        setError(resolved.error === 'not_found' ? t('auth.group_not_found') : t('login2.server_unreachable'));
        setLoading(false);
        return;
      }

      setClientServerUrl(resolved.url);
      const success = await joinGroupWithCode(resolved.url, inviteCode.trim(), joinGroupName.trim());

      if (success) {
        await saveGroupToFirebase(joinGroupName.trim(), 'user');
        const mode = resolved.local ? 'server' : 'client';
        onLoginComplete?.(mode);
      }
    } catch (err) {
      setError(err.message || 'Join failed');
    } finally {
      setLoading(false);
    }
  }, [joinGroupName, inviteCode, localGroupName, port, joinGroupWithCode, saveGroupToFirebase, onLoginComplete, t]);

  const handleCreateServer = useCallback(async (e) => {
    e.preventDefault();
    setError('');
    if (!newGroupName.trim()) { setError(t('validation.group_name_required')); return; }

    if (!isElectron) {
      setError(t('login2.create_requires_electron'));
      return;
    }

    setLoading(true);
    try {
      if (!localServerReady) {
        await window.electron?.server?.start({ port });
        await new Promise(r => setTimeout(r, 2000));
      }

      const baseUrl = `http://localhost:${port}`;
      setClientServerUrl(baseUrl);

      try {
        await window.electron?.pipeline?.setTier(selectedTier);
      } catch { /* ignore */ }

      // Get Firebase ID token for server init
      const idToken = await getIdToken();
      if (!idToken) {
        setError('Firebase authentication required');
        setLoading(false);
        return;
      }

      await firebaseInitServer(baseUrl, {
        group_name: newGroupName.trim(),
        id_token: idToken,
      });

      // Connect to the newly created server
      const success = await connectToGroup(baseUrl, newGroupName.trim());

      if (success) {
        await saveGroupToFirebase(newGroupName.trim(), 'admin');
        setLocalGroupName(newGroupName.trim());
        setLocalServerReady(true);
        onLoginComplete?.('server');
      }
    } catch (err) {
      setError(err.message || 'Server creation failed');
    } finally {
      setLoading(false);
    }
  }, [newGroupName, selectedTier, port, localServerReady, connectToGroup, saveGroupToFirebase, onLoginComplete, t]);

  const handleFullLogout = useCallback(async () => {
    await fullLogout();
    setStage('auth');
    setMyGroups([]);
    setEmail('');
    setPassword('');
  }, [fullLogout]);

  const displayError = error || authError;
  const togglePassword = () => setShowPassword(!showPassword);
  const pwProps = { showPassword, onTogglePassword: togglePassword };

  // -----------------------------------------------------------------------
  // Stage 1: Firebase Auth
  // -----------------------------------------------------------------------
  const renderAuthStage = () => (
    <>
      {authView === 'login' ? (
        <form onSubmit={handleFirebaseLogin} className="space-y-4">
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

          <div className="flex items-center gap-3 text-xs text-zinc-500">
            <div className="flex-1 border-t border-zinc-700/50" />
            <span>{t('login2.or')}</span>
            <div className="flex-1 border-t border-zinc-700/50" />
          </div>

          <button
            type="button"
            onClick={() => { setError(''); setAuthView('register'); }}
            className="w-full py-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700/50 rounded-lg
              hover:border-zinc-600 transition-colors flex items-center justify-center gap-1.5"
          >
            <UserPlus size={13} />
            {t('auth.create_account')}
          </button>
        </form>
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
  // Stage 2: Group Selection
  // -----------------------------------------------------------------------
  const renderGroupStage = () => (
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

      {/* Group sub-views */}
      {groupView === 'list' && renderGroupList()}
      {groupView === 'join' && renderJoinGroup()}
      {groupView === 'create' && renderCreateGroup()}
    </div>
  );

  // ── Group List ──
  const renderGroupList = () => (
    <div className="space-y-3">
      <p className="text-xs font-medium text-zinc-400">{t('login2.my_groups')}</p>

      {groupsLoading ? (
        <div className="flex items-center justify-center py-6">
          <Loader2 size={18} className="animate-spin text-zinc-500" />
        </div>
      ) : myGroups.length > 0 ? (
        <div className="space-y-2">
          {myGroups.map(g => (
            <button
              key={g.key}
              onClick={() => handleConnectGroup(g.group_name)}
              disabled={!!connectingGroup}
              className="w-full p-3 bg-zinc-800/60 border border-zinc-700/50 rounded-lg
                hover:border-zinc-600 hover:bg-zinc-800/80 transition-colors
                flex items-center gap-3 disabled:opacity-50 text-left"
            >
              <div className="w-8 h-8 rounded-lg bg-blue-600/20 flex items-center justify-center shrink-0">
                <Users size={16} className="text-blue-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-zinc-200 font-medium truncate">{g.group_name}</p>
                <p className="text-[10px] text-zinc-500">{g.role}</p>
              </div>
              {connectingGroup === g.group_name ? (
                <Loader2 size={14} className="animate-spin text-blue-400 shrink-0" />
              ) : (
                <ArrowLeft size={14} className="text-zinc-600 rotate-180 shrink-0" />
              )}
            </button>
          ))}
        </div>
      ) : (
        <div className="text-center py-6">
          <Users size={24} className="mx-auto text-zinc-600 mb-2" />
          <p className="text-xs text-zinc-500">{t('login2.no_groups')}</p>
        </div>
      )}

      {/* Local server shortcut */}
      {localGroupName && (
        <button
          onClick={() => handleConnectGroup(localGroupName)}
          disabled={!!connectingGroup}
          className="w-full p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg
            hover:bg-emerald-500/15 transition-colors flex items-center gap-3 disabled:opacity-50"
        >
          <Server size={16} className="text-emerald-400 shrink-0" />
          <div className="flex-1 text-left min-w-0">
            <p className="text-xs text-emerald-300 font-medium truncate">{localGroupName}</p>
            <p className="text-[10px] text-zinc-500">{t('login2.local_server')}</p>
          </div>
          {connectingGroup === localGroupName ? (
            <Loader2 size={14} className="animate-spin text-emerald-400" />
          ) : (
            <CheckCircle size={14} className="text-emerald-500/50 shrink-0" />
          )}
        </button>
      )}

      {/* Action buttons */}
      <div className="flex gap-2 pt-1">
        <button
          type="button"
          onClick={() => { setError(''); setGroupView('join'); }}
          className="flex-1 py-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700/50 rounded-lg
            hover:border-zinc-600 transition-colors flex items-center justify-center gap-1.5"
        >
          <Key size={13} />
          {t('login2.join_with_code')}
        </button>
        {isElectron && (
          <button
            type="button"
            onClick={() => { setError(''); setGroupView('create'); }}
            className="flex-1 py-2 text-xs text-zinc-400 hover:text-zinc-200 border border-zinc-700/50 rounded-lg
              hover:border-zinc-600 transition-colors flex items-center justify-center gap-1.5"
          >
            <Plus size={13} />
            {t('login2.create_group')}
          </button>
        )}
      </div>
    </div>
  );

  // ── Join Group ──
  const renderJoinGroup = () => (
    <form onSubmit={handleJoinGroup} className="space-y-4">
      <button
        type="button"
        onClick={() => { setError(''); setGroupView('list'); }}
        className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 mb-1"
      >
        <ArrowLeft size={13} /> {t('login2.back_to_groups')}
      </button>

      <InputField
        icon={Server}
        label={t('login2.group_name')}
        value={joinGroupName}
        onChange={setJoinGroupName}
        placeholder={t('login2.group_name_placeholder')}
        autoFocus
        disabled={loading}
      />
      <InputField
        icon={Key}
        label={t('login2.invite_code')}
        value={inviteCode}
        onChange={setInviteCode}
        placeholder="ABC123..."
        disabled={loading}
      />

      <SubmitButton loading={loading}>
        <Users size={15} />
        {t('login2.join_group')}
      </SubmitButton>
    </form>
  );

  // ── Create Group (Electron) ──
  const renderCreateGroup = () => (
    <form onSubmit={handleCreateServer} className="space-y-4">
      <button
        type="button"
        onClick={() => { setError(''); setGroupView('list'); }}
        className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300 mb-1"
      >
        <ArrowLeft size={13} /> {t('login2.back_to_groups')}
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
            {stage === 'auth' ? t('auth.subtitle') : t('login2.select_group')}
          </p>
        </div>

        {/* Card */}
        <div className="bg-zinc-800/40 border border-zinc-700/40 rounded-xl p-5 shadow-xl">
          {/* View title */}
          <h2 className="text-sm font-semibold text-zinc-200 mb-4">
            {stage === 'auth' && (
              authView === 'login' ? t('auth.login') : t('auth.create_account')
            )}
            {stage === 'groups' && (
              groupView === 'list' ? t('login2.my_groups') :
              groupView === 'join' ? t('login2.join_group') :
              t('login2.create_group')
            )}
          </h2>

          {/* Error display */}
          {displayError && (
            <div className="mb-4 p-2.5 bg-red-500/10 border border-red-500/30 rounded-lg flex items-start gap-2">
              <AlertCircle size={14} className="text-red-400 mt-0.5 shrink-0" />
              <p className="text-xs text-red-300">{displayError}</p>
            </div>
          )}

          {/* Local server starting */}
          {localServerStarting && (
            <div className="mb-4 p-2.5 bg-blue-500/10 border border-blue-500/30 rounded-lg flex items-center gap-2">
              <Loader2 size={14} className="text-blue-400 animate-spin" />
              <p className="text-xs text-blue-300">{t('login2.detecting_local')}</p>
            </div>
          )}

          {/* Stage content */}
          {stage === 'auth' && renderAuthStage()}
          {stage === 'groups' && renderGroupStage()}
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
