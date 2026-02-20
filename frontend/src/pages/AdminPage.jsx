/**
 * AdminPage — user management, invite codes, job queue monitoring.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../i18n';
import { useAuth } from '../contexts/AuthContext';
import {
  listUsers, updateUser, deleteUser,
  listInviteCodes, createInviteCode,
  cleanupStaleJobs, getJobStats,
  browseFolders, scanFolder,
  listWorkerTokens, createWorkerToken, revokeWorkerToken,
  listWorkerSessions, stopWorkerSession, blockWorkerSession,
} from '../api/admin';
import {
  Users, Key, Activity, FolderSearch, Terminal, Server,
  Shield, ShieldOff, Trash2, Copy, Plus, Square, Ban,
  RefreshCw, CheckCircle, XCircle, AlertTriangle,
  Folder, FolderOpen, ChevronRight, ArrowUp, Play, Loader2,
} from 'lucide-react';

export default function AdminPage() {
  const { t } = useLocale();
  const { user: currentUser } = useAuth();
  const [activeTab, setActiveTab] = useState('discover');

  const tabs = [
    { id: 'discover', label: t('admin.tab_discover'), icon: FolderSearch },
    { id: 'queue', label: t('admin.tab_queue'), icon: Activity },
    { id: 'workers', label: t('admin.tab_workers'), icon: Server },
    { id: 'users', label: t('admin.tab_users'), icon: Users },
    { id: 'invites', label: t('admin.tab_invites'), icon: Key },
    { id: 'worker_tokens', label: t('admin.tab_worker_tokens'), icon: Terminal },
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
        {activeTab === 'discover' && <DiscoverPanel />}
        {activeTab === 'queue' && <QueuePanel />}
        {activeTab === 'workers' && <WorkersPanel />}
        {activeTab === 'users' && <UsersPanel />}
        {activeTab === 'invites' && <InvitesPanel />}
        {activeTab === 'worker_tokens' && <WorkerTokensPanel />}
      </div>
    </div>
  );
}


// ── Workers Panel ────────────────────────────────────────

function WorkersPanel() {
  const { t } = useLocale();
  const [workers, setWorkers] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const data = await listWorkerSessions();
      setWorkers(data.workers || []);
    } catch (e) {
      console.error('Failed to load workers:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, [load]);

  const handleStop = async (id) => {
    if (!confirm(t('admin.worker_confirm_stop'))) return;
    try {
      await stopWorkerSession(id);
      load();
    } catch (e) {
      console.error('Failed to stop worker:', e);
    }
  };

  const handleBlock = async (id) => {
    if (!confirm(t('admin.worker_confirm_block'))) return;
    try {
      await blockWorkerSession(id);
      load();
    } catch (e) {
      console.error('Failed to block worker:', e);
    }
  };

  const onlineCount = workers.filter(w => w.status === 'online').length;
  const totalThroughput = workers.reduce((sum, w) => sum + (w.throughput || 0), 0);
  const totalCompleted = workers.reduce((sum, w) => sum + (w.jobs_completed || 0), 0);

  const statusBadge = (status) => {
    const map = {
      online: 'bg-green-900/50 text-green-300',
      offline: 'bg-gray-700 text-gray-400',
      blocked: 'bg-red-900/50 text-red-300',
    };
    return (
      <span className={`px-2 py-0.5 rounded text-xs ${map[status] || 'bg-gray-700 text-gray-400'}`}>
        {t(`admin.worker_status_${status}`)}
      </span>
    );
  };

  const timeAgo = (isoStr) => {
    if (!isoStr) return '-';
    const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000);
    if (diff < 60) return `${diff}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    return `${Math.floor(diff / 3600)}h`;
  };

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">
          {t('admin.workers_title')}
          {onlineCount > 0 && (
            <span className="ml-2 text-sm font-normal text-green-400">({onlineCount} online)</span>
          )}
        </h2>
        <button
          onClick={load}
          className="p-1.5 rounded hover:bg-gray-600 text-gray-400 hover:text-white"
        >
          <RefreshCw size={14} />
        </button>
      </div>

      {/* Aggregate stats */}
      {onlineCount > 0 && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 text-center">
            <div className="text-xs text-gray-400 mb-0.5">{t('admin.worker_total_speed')}</div>
            <div className="text-xl font-bold text-emerald-400 font-mono">
              {totalThroughput > 0 ? totalThroughput.toFixed(1) : '-'}
              {totalThroughput > 0 && <span className="text-xs font-normal text-emerald-400/60 ml-1">{t('admin.queue_files_per_min')}</span>}
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 text-center">
            <div className="text-xs text-gray-400 mb-0.5">{t('admin.worker_online_count')}</div>
            <div className="text-xl font-bold text-green-400 font-mono">{onlineCount}</div>
          </div>
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3 text-center">
            <div className="text-xs text-gray-400 mb-0.5">{t('admin.worker_total_completed')}</div>
            <div className="text-xl font-bold text-blue-400 font-mono">{totalCompleted}</div>
          </div>
        </div>
      )}

      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400">
              <th className="text-left px-4 py-3">{t('admin.worker_name')}</th>
              <th className="text-left px-4 py-3">{t('auth.username')}</th>
              <th className="text-left px-4 py-3">Status</th>
              <th className="text-left px-4 py-3">{t('admin.worker_capacity')}</th>
              <th className="text-left px-4 py-3">{t('admin.worker_speed')}</th>
              <th className="text-left px-4 py-3">{t('admin.worker_jobs_done')}</th>
              <th className="text-left px-4 py-3">{t('admin.worker_current_task')}</th>
              <th className="text-left px-4 py-3">{t('admin.worker_last_heartbeat')}</th>
              <th className="text-right px-4 py-3"></th>
            </tr>
          </thead>
          <tbody>
            {workers.map((w) => (
              <tr key={w.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                <td className="px-4 py-3 font-medium">
                  <div>{w.worker_name}</div>
                  {w.hostname && <div className="text-xs text-gray-500">{w.hostname}</div>}
                </td>
                <td className="px-4 py-3 text-gray-400">{w.username}</td>
                <td className="px-4 py-3">{statusBadge(w.status)}</td>
                <td className="px-4 py-3">
                  <span className="font-mono text-yellow-300">B:{w.batch_capacity}</span>
                </td>
                <td className="px-4 py-3">
                  {w.throughput > 0 ? (
                    <span className="font-mono text-emerald-400">{w.throughput.toFixed(1)}<span className="text-emerald-400/50 text-xs ml-0.5">/m</span></span>
                  ) : (
                    <span className="text-gray-600">-</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <span className="text-green-400">{w.jobs_completed}</span>
                  {w.jobs_failed > 0 && (
                    <span className="text-red-400 ml-1">/ {w.jobs_failed} fail</span>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-400 text-xs">
                  {w.current_phase ? (
                    <span>
                      <span className="text-blue-300">{w.current_phase}</span>
                      {w.current_file && (
                        <span className="ml-1 text-gray-500 truncate max-w-[120px] inline-block align-bottom">
                          {w.current_file.split(/[/\\]/).pop()}
                        </span>
                      )}
                    </span>
                  ) : '-'}
                </td>
                <td className="px-4 py-3 text-gray-500 text-xs font-mono">
                  {timeAgo(w.last_heartbeat)}
                </td>
                <td className="px-4 py-3 text-right">
                  {w.status === 'online' && (
                    <div className="flex gap-1 justify-end">
                      <button
                        onClick={() => handleStop(w.id)}
                        className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-gray-700 hover:bg-yellow-900/50 text-gray-300 hover:text-yellow-300"
                        title={t('admin.worker_action_stop')}
                      >
                        <Square size={10} />
                        {t('admin.worker_action_stop')}
                      </button>
                      <button
                        onClick={() => handleBlock(w.id)}
                        className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-gray-700 hover:bg-red-900/50 text-gray-300 hover:text-red-300"
                        title={t('admin.worker_action_block')}
                      >
                        <Ban size={10} />
                        {t('admin.worker_action_block')}
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {workers.length === 0 && (
          <div className="text-center text-gray-500 py-8 text-sm">{t('admin.workers_empty')}</div>
        )}
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


// ── Discover Panel ──────────────────────────────────────

function DiscoverPanel() {
  const { t } = useLocale();
  const [currentPath, setCurrentPath] = useState('/');
  const [pathInput, setPathInput] = useState('/');
  const [browseData, setBrowseData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [scanning, setScanning] = useState(null); // folder name being scanned
  const [scanResult, setScanResult] = useState(null);
  const [priority, setPriority] = useState(0);
  const [error, setError] = useState('');

  const browse = useCallback(async (path) => {
    setLoading(true);
    setError('');
    setScanResult(null);
    try {
      const data = await browseFolders(path);
      setBrowseData(data);
      setCurrentPath(data.current);
      setPathInput(data.current);
    } catch (e) {
      setError(e.detail || e.message || 'Browse failed');
      setBrowseData(null);
    }
    setLoading(false);
  }, []);

  useEffect(() => { browse('/'); }, [browse]);

  const handleBrowse = () => browse(pathInput);
  const handleKeyDown = (e) => { if (e.key === 'Enter') handleBrowse(); };

  const handleNavigate = (dirName) => {
    const next = currentPath.endsWith('/') ? currentPath + dirName : currentPath + '/' + dirName;
    browse(next);
  };

  const handleGoUp = () => {
    if (browseData?.parent) browse(browseData.parent);
  };

  const handleScan = async (folderPath) => {
    setScanning(folderPath);
    setScanResult(null);
    try {
      const data = await scanFolder(folderPath, priority);
      setScanResult(data);
    } catch (e) {
      setScanResult({ success: false, error: e.detail || e.message });
    }
    setScanning(null);
  };

  const handleScanCurrent = () => handleScan(currentPath);

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">{t('admin.discover_title')}</h2>

      {/* Path input */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
        <div className="flex gap-2 mb-3">
          <input
            type="text"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="/mnt/nas/assets"
            className="flex-1 px-3 py-2 bg-gray-900 border border-gray-600 rounded-lg text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none font-mono"
          />
          <button
            onClick={handleBrowse}
            disabled={loading}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 rounded-lg text-sm text-white"
          >
            {t('admin.discover_browse')}
          </button>
          <button
            onClick={handleScanCurrent}
            disabled={!!scanning || loading}
            className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 rounded-lg text-sm text-white"
          >
            {scanning === currentPath ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {t('admin.discover_scan_all')}
          </button>
        </div>

        {/* Priority */}
        <div className="flex items-center gap-2 text-xs text-gray-400">
          <span>{t('admin.discover_priority')}:</span>
          <input
            type="number"
            min={0}
            max={10}
            value={priority}
            onChange={(e) => setPriority(parseInt(e.target.value) || 0)}
            className="w-16 px-2 py-1 bg-gray-900 border border-gray-600 rounded text-sm text-white"
          />
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 mb-4 bg-red-900/30 border border-red-800 rounded-lg text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Scan result */}
      {scanResult && (
        <div className={`p-3 mb-4 rounded-lg text-xs border ${
          scanResult.success
            ? 'bg-green-900/30 border-green-800 text-green-400'
            : 'bg-red-900/30 border-red-800 text-red-400'
        }`}>
          {scanResult.success
            ? t('admin.discover_scan_result', {
                discovered: scanResult.discovered,
                jobs: scanResult.jobs_created,
                skipped: scanResult.skipped,
              })
            : scanResult.error}
        </div>
      )}

      {/* Directory listing */}
      {loading ? (
        <div className="text-gray-400 text-sm flex items-center gap-2">
          <Loader2 size={14} className="animate-spin" /> {t('status.loading')}
        </div>
      ) : browseData && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          {/* Current path header */}
          <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-700 bg-gray-800/50">
            <button
              onClick={handleGoUp}
              disabled={!browseData.parent}
              className="p-1 rounded hover:bg-gray-600 disabled:opacity-30 text-gray-400"
              title="Parent directory"
            >
              <ArrowUp size={16} />
            </button>
            <code className="text-xs text-gray-300 font-mono flex-1 truncate">{currentPath}</code>
            {browseData.files_count > 0 && (
              <span className="text-xs text-gray-500">
                {browseData.files_count} {t('admin.discover_files_here')}
              </span>
            )}
          </div>

          {/* Directories */}
          {browseData.dirs.length === 0 && browseData.files_count === 0 ? (
            <div className="text-center text-gray-500 py-8 text-sm">{t('admin.discover_empty')}</div>
          ) : (
            <div className="divide-y divide-gray-700/50">
              {browseData.dirs.map((dir) => {
                const fullPath = currentPath.endsWith('/')
                  ? currentPath + dir.name
                  : currentPath + '/' + dir.name;
                return (
                  <div
                    key={dir.name}
                    className="flex items-center gap-3 px-4 py-2.5 hover:bg-gray-700/30 group"
                  >
                    <Folder size={16} className="text-yellow-500 flex-shrink-0" />
                    <button
                      onClick={() => handleNavigate(dir.name)}
                      className="flex-1 text-left text-sm text-gray-200 hover:text-white truncate"
                    >
                      {dir.name}/
                    </button>
                    <span className="text-xs text-gray-500 flex-shrink-0">
                      {dir.files_count > 0 ? `${dir.files_count} files` : ''}
                    </span>
                    <button
                      onClick={() => handleScan(fullPath)}
                      disabled={!!scanning}
                      className="flex items-center gap-1 px-2.5 py-1 bg-gray-700 hover:bg-blue-600 rounded text-xs text-gray-300 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-30"
                    >
                      {scanning === fullPath ? (
                        <Loader2 size={12} className="animate-spin" />
                      ) : (
                        <Play size={12} />
                      )}
                      Scan
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// ── Worker Tokens Panel ──────────────────────────────────

function WorkerTokensPanel() {
  const { t } = useLocale();
  const [tokens, setTokens] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tokenName, setTokenName] = useState('');
  const [expiresDays, setExpiresDays] = useState(30);
  const [creating, setCreating] = useState(false);
  const [newToken, setNewToken] = useState(null); // just-created token info
  const [copiedField, setCopiedField] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listWorkerTokens();
      setTokens(Array.isArray(data) ? data : []);
    } catch (e) {
      console.error('Failed to load worker tokens:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleCreate = async () => {
    if (!tokenName.trim()) return;
    setCreating(true);
    try {
      const data = await createWorkerToken({ name: tokenName.trim(), expires_in_days: expiresDays });
      setNewToken(data);
      setTokenName('');
      load();
    } catch (e) {
      console.error('Failed to create worker token:', e);
    }
    setCreating(false);
  };

  const handleRevoke = async (tokenId) => {
    if (!confirm(t('admin.worker_token_confirm_revoke'))) return;
    try {
      await revokeWorkerToken(tokenId);
      load();
    } catch (e) {
      console.error('Failed to revoke token:', e);
    }
  };

  const handleCopy = (text, field) => {
    navigator.clipboard.writeText(text);
    setCopiedField(field);
    setTimeout(() => setCopiedField(null), 2000);
  };

  const serverUrl = window.location.origin;
  const runCommand = newToken?.token
    ? `python -m backend.worker.worker_daemon --server ${serverUrl} --token ${newToken.token}`
    : '';

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">{t('admin.worker_tokens_title')}</h2>

      {/* Create form */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
        <div className="flex gap-3 items-end flex-wrap">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-xs text-gray-400 mb-1">{t('admin.worker_token_name')}</label>
            <input
              type="text"
              value={tokenName}
              onChange={(e) => setTokenName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && tokenName.trim()) handleCreate(); }}
              placeholder={t('admin.worker_token_name_placeholder')}
              className="w-full px-2 py-1.5 bg-gray-900 border border-gray-600 rounded text-sm text-white placeholder-gray-500"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('admin.worker_token_expires')}</label>
            <input
              type="number"
              min={1}
              max={365}
              value={expiresDays}
              onChange={(e) => setExpiresDays(parseInt(e.target.value) || 30)}
              className="w-20 px-2 py-1.5 bg-gray-900 border border-gray-600 rounded text-sm text-white"
            />
          </div>
          <button
            onClick={handleCreate}
            disabled={creating || !tokenName.trim()}
            className="flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white rounded text-sm"
          >
            <Plus size={14} />
            {t('admin.create_worker_token')}
          </button>
        </div>
      </div>

      {/* New token display (shown once) */}
      {newToken?.token && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 mb-4">
          <div className="flex items-center gap-2 mb-3 text-yellow-300 text-sm font-medium">
            <AlertTriangle size={16} />
            {t('admin.worker_token_warning')}
          </div>

          {/* Token value */}
          <div className="mb-3">
            <label className="block text-xs text-gray-400 mb-1">Token</label>
            <div className="flex gap-2">
              <code className="flex-1 px-3 py-2 bg-gray-900 rounded text-xs font-mono text-green-300 break-all select-all">
                {newToken.token}
              </code>
              <button
                onClick={() => handleCopy(newToken.token, 'token')}
                className="flex items-center gap-1 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs text-white flex-shrink-0"
              >
                {copiedField === 'token' ? <CheckCircle size={12} className="text-green-400" /> : <Copy size={12} />}
                {copiedField === 'token' ? t('admin.worker_token_copied') : t('admin.worker_token_copy')}
              </button>
            </div>
          </div>

          {/* Run command */}
          <div>
            <label className="block text-xs text-gray-400 mb-1">{t('admin.worker_token_command')}</label>
            <div className="flex gap-2">
              <code className="flex-1 px-3 py-2 bg-gray-900 rounded text-xs font-mono text-blue-300 break-all select-all">
                {runCommand}
              </code>
              <button
                onClick={() => handleCopy(runCommand, 'command')}
                className="flex items-center gap-1 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs text-white flex-shrink-0"
              >
                {copiedField === 'command' ? <CheckCircle size={12} className="text-green-400" /> : <Copy size={12} />}
                {copiedField === 'command' ? t('admin.worker_token_copied') : t('admin.worker_token_copy')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Token list */}
      {loading ? (
        <div className="text-gray-400 text-sm">{t('status.loading')}</div>
      ) : (
        <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 text-gray-400">
                <th className="text-left px-4 py-3">ID</th>
                <th className="text-left px-4 py-3">{t('admin.worker_token_name')}</th>
                <th className="text-left px-4 py-3">{t('admin.user_active')}</th>
                <th className="text-left px-4 py-3">{t('admin.worker_token_created')}</th>
                <th className="text-left px-4 py-3">{t('admin.worker_token_last_used')}</th>
                <th className="text-right px-4 py-3"></th>
              </tr>
            </thead>
            <tbody>
              {tokens.map((tk) => (
                <tr key={tk.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                  <td className="px-4 py-3 text-gray-500">{tk.id}</td>
                  <td className="px-4 py-3 font-medium">{tk.name}</td>
                  <td className="px-4 py-3">
                    {tk.is_active ? (
                      <span className="px-2 py-0.5 rounded text-xs bg-green-900/50 text-green-300">
                        {t('admin.worker_token_active')}
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 rounded text-xs bg-red-900/50 text-red-300">
                        {t('admin.worker_token_revoked')}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">
                    {tk.created_at ? new Date(tk.created_at).toLocaleDateString() : '-'}
                  </td>
                  <td className="px-4 py-3 text-gray-500 text-xs">
                    {tk.last_used_at ? new Date(tk.last_used_at).toLocaleString() : t('admin.worker_token_never_used')}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {tk.is_active && (
                      <button
                        onClick={() => handleRevoke(tk.id)}
                        className="p-1.5 rounded hover:bg-red-900/50 text-gray-400 hover:text-red-400"
                        title={t('admin.worker_token_revoke')}
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {tokens.length === 0 && (
            <div className="text-center text-gray-500 py-8 text-sm">No worker tokens</div>
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
    try {
      const data = await getJobStats();
      setStats(data);
    } catch (e) {
      console.error('Failed to load queue stats:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, [load]);

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

  const throughput = stats?.throughput ?? 0;
  const remaining = (stats?.pending ?? 0) + (stats?.assigned ?? 0) + (stats?.processing ?? 0);
  const etaMin = throughput > 0 ? Math.ceil(remaining / throughput) : null;

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

      {/* Throughput banner */}
      {throughput > 0 && (
        <div className="bg-emerald-900/30 border border-emerald-700/50 rounded-lg p-4 mb-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div>
              <div className="text-xs text-emerald-400/70 mb-0.5">{t('admin.queue_throughput')}</div>
              <div className="text-2xl font-bold text-emerald-400 font-mono">
                {throughput.toFixed(1)}<span className="text-sm font-normal text-emerald-400/60 ml-1">{t('admin.queue_files_per_min')}</span>
              </div>
            </div>
            {etaMin !== null && remaining > 0 && (
              <div className="border-l border-emerald-700/50 pl-4">
                <div className="text-xs text-emerald-400/70 mb-0.5">{t('admin.queue_eta')}</div>
                <div className="text-lg font-bold text-emerald-300 font-mono">
                  {etaMin < 60 ? `${etaMin}m` : `${Math.floor(etaMin / 60)}h ${etaMin % 60}m`}
                </div>
              </div>
            )}
            {remaining > 0 && (
              <div className="border-l border-emerald-700/50 pl-4">
                <div className="text-xs text-emerald-400/70 mb-0.5">{t('admin.queue_remaining')}</div>
                <div className="text-lg font-bold text-gray-300 font-mono">{remaining}</div>
              </div>
            )}
          </div>
          <div className="text-xs text-gray-500">
            {t('admin.queue_recent_window', { count_1m: stats?.recent_1min ?? 0, count_5m: stats?.recent_5min ?? 0 })}
          </div>
        </div>
      )}

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
            <span>{((stats.completed / stats.total) * 100).toFixed(1)}% {t('admin.queue_completed').toLowerCase()}</span>
            <span>{stats.total} {t('admin.queue_total').toLowerCase()}</span>
          </div>
        </div>
      )}
    </div>
  );
}
