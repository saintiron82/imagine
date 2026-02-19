/**
 * AdminPage — user management, invite codes, job queue monitoring.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../i18n';
import { useAuth } from '../contexts/AuthContext';
import {
  listUsers, updateUser, deleteUser,
  listInviteCodes, createInviteCode,
  cleanupStaleJobs,
} from '../api/admin';
import { getJobStats } from '../api/stats';
import {
  Users, Key, Activity,
  Shield, ShieldOff, Trash2, Copy, Plus,
  RefreshCw, CheckCircle, XCircle,
} from 'lucide-react';

export default function AdminPage() {
  const { t } = useLocale();
  const { user: currentUser } = useAuth();
  const [activeTab, setActiveTab] = useState('users');

  const tabs = [
    { id: 'users', label: t('admin.tab_users'), icon: Users },
    { id: 'invites', label: t('admin.tab_invites'), icon: Key },
    { id: 'queue', label: t('admin.tab_queue'), icon: Activity },
  ];

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      {/* Tab bar */}
      <div className="flex border-b border-gray-700 px-4 pt-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === id
                ? 'border-blue-500 text-white'
                : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'users' && <UsersPanel />}
        {activeTab === 'invites' && <InvitesPanel />}
        {activeTab === 'queue' && <QueuePanel />}
      </div>
    </div>
  );
}


// ── Users Panel ──────────────────────────────────────────

function UsersPanel() {
  const { t } = useLocale();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listUsers();
      setUsers(data.users || []);
    } catch (e) {
      console.error('Failed to load users:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleToggleActive = async (u) => {
    await updateUser(u.id, { is_active: !u.is_active });
    load();
  };

  const handleToggleRole = async (u) => {
    await updateUser(u.id, { role: u.role === 'admin' ? 'user' : 'admin' });
    load();
  };

  const handleDelete = async (u) => {
    if (!confirm(t('admin.confirm_delete_user'))) return;
    await deleteUser(u.id);
    load();
  };

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">{t('admin.users_title')}</h2>
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400">
              <th className="text-left px-4 py-3">ID</th>
              <th className="text-left px-4 py-3">{t('auth.username')}</th>
              <th className="text-left px-4 py-3">{t('auth.email')}</th>
              <th className="text-left px-4 py-3">{t('admin.user_role')}</th>
              <th className="text-left px-4 py-3">{t('admin.user_active')}</th>
              <th className="text-left px-4 py-3">{t('admin.user_created')}</th>
              <th className="text-right px-4 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                <td className="px-4 py-3 text-gray-500">{u.id}</td>
                <td className="px-4 py-3 font-medium">{u.username}</td>
                <td className="px-4 py-3 text-gray-400">{u.email}</td>
                <td className="px-4 py-3">
                  <span className={`px-2 py-0.5 rounded text-xs ${
                    u.role === 'admin' ? 'bg-purple-900/50 text-purple-300' : 'bg-gray-700 text-gray-300'
                  }`}>
                    {u.role}
                  </span>
                </td>
                <td className="px-4 py-3">
                  {u.is_active ? (
                    <CheckCircle size={16} className="text-green-400" />
                  ) : (
                    <XCircle size={16} className="text-red-400" />
                  )}
                </td>
                <td className="px-4 py-3 text-gray-500 text-xs">
                  {u.created_at ? new Date(u.created_at).toLocaleDateString() : '-'}
                </td>
                <td className="px-4 py-3 text-right">
                  <div className="flex gap-1 justify-end">
                    <button
                      onClick={() => handleToggleActive(u)}
                      className="p-1.5 rounded hover:bg-gray-600 text-gray-400 hover:text-white"
                      title={u.is_active ? t('admin.action_deactivate') : t('admin.action_activate')}
                    >
                      {u.is_active ? <ShieldOff size={14} /> : <Shield size={14} />}
                    </button>
                    <button
                      onClick={() => handleToggleRole(u)}
                      className="p-1.5 rounded hover:bg-gray-600 text-gray-400 hover:text-white"
                      title={u.role === 'admin' ? t('admin.action_make_user') : t('admin.action_make_admin')}
                    >
                      <Shield size={14} />
                    </button>
                    <button
                      onClick={() => handleDelete(u)}
                      className="p-1.5 rounded hover:bg-red-900/50 text-gray-400 hover:text-red-400"
                      title={t('admin.action_delete_user')}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {users.length === 0 && (
          <div className="text-center text-gray-500 py-8 text-sm">No users found</div>
        )}
      </div>
    </div>
  );
}


// ── Invites Panel ────────────────────────────────────────

function InvitesPanel() {
  const { t } = useLocale();
  const [codes, setCodes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [maxUses, setMaxUses] = useState(1);
  const [expiresDays, setExpiresDays] = useState(7);
  const [note, setNote] = useState('');
  const [creating, setCreating] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listInviteCodes();
      setCodes(data.codes || []);
    } catch (e) {
      console.error('Failed to load invite codes:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleCreate = async () => {
    setCreating(true);
    try {
      await createInviteCode({ max_uses: maxUses, expires_days: expiresDays, note });
      setNote('');
      load();
    } catch (e) {
      console.error('Failed to create invite:', e);
    }
    setCreating(false);
  };

  const handleCopy = (code) => {
    navigator.clipboard.writeText(code);
    setCopiedId(code);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">{t('admin.invites_title')}</h2>

      {/* Create form */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
        <div className="flex gap-3 items-end flex-wrap">
          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('admin.invite_max_uses')}</label>
            <input
              type="number"
              min={1}
              max={100}
              value={maxUses}
              onChange={(e) => setMaxUses(parseInt(e.target.value) || 1)}
              className="w-20 px-2 py-1.5 bg-gray-900 border border-gray-600 rounded text-sm text-white"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('admin.invite_expires_days')}</label>
            <input
              type="number"
              min={1}
              max={365}
              value={expiresDays}
              onChange={(e) => setExpiresDays(parseInt(e.target.value) || 7)}
              className="w-20 px-2 py-1.5 bg-gray-900 border border-gray-600 rounded text-sm text-white"
            />
          </div>
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs text-gray-400 mb-1">{t('admin.invite_note')}</label>
            <input
              type="text"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Optional note..."
              className="w-full px-2 py-1.5 bg-gray-900 border border-gray-600 rounded text-sm text-white placeholder-gray-500"
            />
          </div>
          <button
            onClick={handleCreate}
            disabled={creating}
            className="flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white rounded text-sm"
          >
            <Plus size={14} />
            {t('admin.create_invite')}
          </button>
        </div>
      </div>

      {/* Codes list */}
      {loading ? (
        <div className="text-gray-400 text-sm">{t('status.loading')}</div>
      ) : (
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-gray-400">
                <th className="text-left px-4 py-3">{t('admin.invite_code')}</th>
                <th className="text-left px-4 py-3">{t('admin.invite_uses')}</th>
                <th className="text-left px-4 py-3">{t('admin.invite_note')}</th>
                <th className="text-left px-4 py-3">{t('admin.invite_expires_days')}</th>
                <th className="text-right px-4 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {codes.map((c) => {
                const isExpired = c.expires_at && new Date(c.expires_at) < new Date();
                const isExhausted = c.use_count >= c.max_uses;
                return (
                  <tr key={c.code} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="px-4 py-3">
                      <code className="font-mono text-xs bg-gray-900 px-2 py-1 rounded">{c.code}</code>
                    </td>
                    <td className="px-4 py-3 text-gray-400">
                      {c.use_count} / {c.max_uses}
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-xs">{c.note || '-'}</td>
                    <td className="px-4 py-3 text-xs">
                      {isExpired ? (
                        <span className="text-red-400">{t('admin.invite_expired')}</span>
                      ) : isExhausted ? (
                        <span className="text-yellow-400">Full</span>
                      ) : (
                        <span className="text-green-400">
                          {c.expires_at ? new Date(c.expires_at).toLocaleDateString() : '-'}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => handleCopy(c.code)}
                        className="p-1.5 rounded hover:bg-gray-600 text-gray-400 hover:text-white"
                        title="Copy"
                      >
                        {copiedId === c.code ? <CheckCircle size={14} className="text-green-400" /> : <Copy size={14} />}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {codes.length === 0 && (
            <div className="text-center text-gray-500 py-8 text-sm">No invite codes</div>
          )}
        </div>
      )}
    </div>
  );
}


// ── Queue Panel ──────────────────────────────────────────

function QueuePanel() {
  const { t } = useLocale();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [cleanupMsg, setCleanupMsg] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getJobStats();
      setStats(data);
    } catch (e) {
      console.error('Failed to load queue stats:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleCleanup = async () => {
    try {
      const data = await cleanupStaleJobs();
      setCleanupMsg(t('admin.cleanup_result', { count: data.reassigned }));
      load();
      setTimeout(() => setCleanupMsg(''), 5000);
    } catch (e) {
      console.error('Cleanup failed:', e);
    }
  };

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  const statItems = [
    { key: 'pending', label: t('admin.queue_pending'), color: 'bg-yellow-500' },
    { key: 'assigned', label: t('admin.queue_assigned'), color: 'bg-blue-500' },
    { key: 'processing', label: t('admin.queue_processing'), color: 'bg-cyan-500' },
    { key: 'completed', label: t('admin.queue_completed'), color: 'bg-green-500' },
    { key: 'failed', label: t('admin.queue_failed'), color: 'bg-red-500' },
  ];

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">{t('admin.queue_title')}</h2>
        <div className="flex items-center gap-2">
          {cleanupMsg && <span className="text-xs text-green-400">{cleanupMsg}</span>}
          <button
            onClick={handleCleanup}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300"
          >
            <RefreshCw size={12} />
            {t('admin.cleanup_stale')}
          </button>
          <button
            onClick={load}
            className="p-1.5 rounded hover:bg-gray-600 text-gray-400 hover:text-white"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        {statItems.map(({ key, label, color }) => (
          <div key={key} className="bg-gray-800 rounded-lg border border-gray-700 p-4 text-center">
            <div className="flex items-center justify-center gap-2 mb-1">
              <div className={`w-2 h-2 rounded-full ${color}`} />
              <span className="text-xs text-gray-400">{label}</span>
            </div>
            <div className="text-2xl font-bold">{stats?.[key] ?? 0}</div>
          </div>
        ))}
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 text-center">
          <div className="text-xs text-gray-400 mb-1">{t('admin.queue_total')}</div>
          <div className="text-2xl font-bold text-blue-400">{stats?.total ?? 0}</div>
        </div>
      </div>

      {/* Progress bar */}
      {stats && stats.total > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
          <div className="flex h-4 rounded-full overflow-hidden bg-gray-900">
            {statItems.map(({ key, color }) => {
              const pct = ((stats[key] || 0) / stats.total) * 100;
              if (pct === 0) return null;
              return (
                <div
                  key={key}
                  className={`${color} transition-all duration-300`}
                  style={{ width: `${pct}%` }}
                  title={`${key}: ${stats[key]}`}
                />
              );
            })}
          </div>
          <div className="flex justify-between mt-2 text-xs text-gray-500">
            <span>{((stats.completed / stats.total) * 100).toFixed(1)}% complete</span>
            <span>{stats.total} total</span>
          </div>
        </div>
      )}
    </div>
  );
}
