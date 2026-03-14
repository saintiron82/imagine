/**
 * AdminPage — user management, invite codes, job queue monitoring.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../i18n';
import { useAuth } from '../contexts/AuthContext';
import {
  cleanupStaleJobs, cleanupQueue, getJobStats,
  browseFolders, scanFolder,
  listWorkerSessions, stopWorkerSession, blockWorkerSession,
  updateWorkerConfig, updateGlobalProcessingMode,
  getAutoProcessing, updateAutoProcessing,
  getEmbeddedWorker, updateEmbeddedWorker,
  listMembers, updateMemberRole, removeMember, deactivateMember, activateMember,
  getThumbnailStats,
  getWorkRequests, getWorkRequestDetail, pauseWorkRequest, resumeWorkRequest, cancelWorkRequest,
  runRecoveryScan,
} from '../api/admin';
import {
  Users, Key, Activity, FolderSearch, Server,
  Shield, ShieldOff, Trash2, Copy, Plus, Square, Ban, UserCheck,
  RefreshCw, CheckCircle, XCircle, AlertTriangle,
  Folder, FolderOpen, ChevronRight, ArrowUp, Play, Loader2,
  Tag, ChevronDown, Pencil, AlertOctagon, Image,
  Pause, CircleX,
} from 'lucide-react';
import { listDomains, getDomainDetail, getActiveDomainConfig, setActiveDomain, saveDomainYaml } from '../services/bridge';

export default function AdminPage() {
  const { t } = useLocale();
  const { user: currentUser } = useAuth();
  const [activeTab, setActiveTab] = useState('discover');

  const tabs = [
    { id: 'discover', label: t('admin.tab_discover'), icon: FolderSearch },
    { id: 'queue', label: t('admin.tab_queue'), icon: Activity },
    { id: 'thumbnails', label: t('admin.tab_thumbnails'), icon: Image },
    { id: 'workers', label: t('admin.tab_workers'), icon: Server },
    { id: 'classification', label: t('admin.tab_classification'), icon: Tag },
    { id: 'members', label: t('admin.tab_members'), icon: UserCheck },
  ];

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      {/* Tab bar */}
      <div className="flex overflow-x-auto border-b border-gray-700 px-4 pt-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap flex-shrink-0 ${
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
        {activeTab === 'thumbnails' && <ThumbnailsPanel />}
        {activeTab === 'workers' && <WorkersPanel />}
        {activeTab === 'classification' && <ClassificationPanel />}
        {activeTab === 'members' && <MembersPanel currentUser={currentUser} />}
      </div>
    </div>
  );
}


// ── Resource Metrics Display ─────────────────────────────

/**
 * Compact resource metrics cell for the workers table.
 * Shows GPU memory bar + temperature, CPU %, and RAM % in a stacked layout.
 * Gracefully handles null/missing data.
 */
function ResourceMetrics({ resources, t }) {
  if (!resources) {
    return <span className="text-gray-600 text-xs">{t('admin.worker_no_metrics')}</span>;
  }

  const {
    gpu_type,
    gpu_memory_used_gb,
    gpu_memory_total_gb,
    gpu_memory_percent,
    gpu_temperature_c,
    cpu_percent,
    memory_percent,
    memory_used_gb,
    memory_total_gb,
  } = resources;

  const hasGpu = gpu_type != null;
  const hasGpuMem = gpu_memory_used_gb != null && gpu_memory_total_gb != null;
  const hasCpu = cpu_percent != null;
  const hasRam = memory_percent != null;

  // Color based on percentage thresholds
  const pctColor = (pct) => {
    if (pct == null) return 'text-gray-500';
    if (pct >= 90) return 'text-red-400';
    if (pct >= 70) return 'text-yellow-400';
    return 'text-emerald-400';
  };

  const barColor = (pct) => {
    if (pct == null) return 'bg-gray-600';
    if (pct >= 90) return 'bg-red-500';
    if (pct >= 70) return 'bg-yellow-500';
    return 'bg-emerald-500';
  };

  const tempColor = (temp) => {
    if (temp == null) return 'text-gray-500';
    if (temp >= 85) return 'text-red-400';
    if (temp >= 70) return 'text-yellow-400';
    return 'text-emerald-400';
  };

  return (
    <div className="flex flex-col gap-1 min-w-[140px]">
      {/* GPU row */}
      {hasGpu && (
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-gray-500 w-7 flex-shrink-0 uppercase">
            {gpu_type === 'cuda' ? t('admin.worker_gpu_type_cuda') : t('admin.worker_gpu_type_mps')}
          </span>
          {hasGpuMem ? (
            <>
              <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden min-w-[40px]">
                <div
                  className={`h-full rounded-full transition-all ${barColor(gpu_memory_percent)}`}
                  style={{ width: `${Math.min(gpu_memory_percent || 0, 100)}%` }}
                />
              </div>
              <span className={`text-[10px] font-mono w-[52px] text-right flex-shrink-0 ${pctColor(gpu_memory_percent)}`}>
                {gpu_memory_used_gb.toFixed(1)}/{gpu_memory_total_gb.toFixed(0)}G
              </span>
            </>
          ) : (
            <span className="text-[10px] text-gray-500">-</span>
          )}
          {gpu_temperature_c != null && (
            <span className={`text-[10px] font-mono flex-shrink-0 ${tempColor(gpu_temperature_c)}`} title="GPU Temperature">
              {gpu_temperature_c}°
            </span>
          )}
        </div>
      )}

      {/* CPU row */}
      {hasCpu && (
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-gray-500 w-7 flex-shrink-0">{t('admin.worker_cpu')}</span>
          <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden min-w-[40px]">
            <div
              className={`h-full rounded-full transition-all ${barColor(cpu_percent)}`}
              style={{ width: `${Math.min(cpu_percent, 100)}%` }}
            />
          </div>
          <span className={`text-[10px] font-mono w-[52px] text-right flex-shrink-0 ${pctColor(cpu_percent)}`}>
            {cpu_percent.toFixed(0)}%
          </span>
        </div>
      )}

      {/* RAM row */}
      {hasRam && (
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] text-gray-500 w-7 flex-shrink-0">{t('admin.worker_memory')}</span>
          <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden min-w-[40px]">
            <div
              className={`h-full rounded-full transition-all ${barColor(memory_percent)}`}
              style={{ width: `${Math.min(memory_percent, 100)}%` }}
            />
          </div>
          <span className={`text-[10px] font-mono w-[52px] text-right flex-shrink-0 ${pctColor(memory_percent)}`}>
            {memory_used_gb != null && memory_total_gb != null
              ? `${memory_used_gb.toFixed(0)}/${memory_total_gb.toFixed(0)}G`
              : `${memory_percent.toFixed(0)}%`
            }
          </span>
        </div>
      )}

      {/* No data at all */}
      {!hasGpu && !hasCpu && !hasRam && (
        <span className="text-gray-600 text-xs">{t('admin.worker_no_metrics')}</span>
      )}
    </div>
  );
}


// ── Workers Panel ────────────────────────────────────────

function WorkersPanel() {
  const { t } = useLocale();
  const [workers, setWorkers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [globalMode, setGlobalMode] = useState('auto');
  const [editingCapacity, setEditingCapacity] = useState(null); // { id, value }
  const [autoProcessing, setAutoProcessing] = useState(true);
  const [restAfterBatch, setRestAfterBatch] = useState(30);
  const [embeddedEnabled, setEmbeddedEnabled] = useState(false);
  const [embeddedStatus, setEmbeddedStatus] = useState({ running: false, status: 'idle', jobs_completed: 0 });

  const load = useCallback(async () => {
    try {
      const data = await listWorkerSessions();
      const all = data.workers || [];
      setWorkers(all.filter(w => w.status === 'online'));
      // Infer global mode from server response or first worker
      if (data.global_processing_mode) {
        setGlobalMode(data.global_processing_mode);
      }
    } catch (e) {
      console.error('Failed to load workers:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    // Load auto_processing config once
    getAutoProcessing().then(data => {
      if (data.enabled != null) setAutoProcessing(data.enabled);
      if (data.rest_after_batch_s != null) setRestAfterBatch(data.rest_after_batch_s);
    }).catch(() => {});
    // Load embedded worker status
    const loadEmbedded = () => {
      getEmbeddedWorker().then(data => {
        if (data.enabled != null) setEmbeddedEnabled(data.enabled);
        setEmbeddedStatus({ running: data.running, status: data.status, jobs_completed: data.jobs_completed || 0 });
      }).catch(() => {});
    };
    loadEmbedded();
    const interval = setInterval(() => { load(); loadEmbedded(); }, 5000);
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

  const handleGlobalMode = async (mode) => {
    try {
      await updateGlobalProcessingMode(mode);
      setGlobalMode(mode);
      load();
    } catch (e) {
      console.error('Failed to update global mode:', e);
    }
  };

  // Per-worker processing_mode override removed — auto-detected by GPU capability.

  const handleCapacitySave = async (sessionId) => {
    if (!editingCapacity) return;
    try {
      await updateWorkerConfig(sessionId, { batch_capacity: editingCapacity.value });
      setEditingCapacity(null);
      load();
    } catch (e) {
      console.error('Failed to update capacity:', e);
    }
  };

  const onlineCount = workers.filter(w => w.status === 'online').length;
  const totalThroughput = workers.reduce((sum, w) => sum + (w.throughput || 0), 0);
  const totalCompleted = workers.reduce((sum, w) => sum + (w.jobs_completed || 0), 0);

  // Combined state badge: online/offline/blocked + worker_state + throttle_level
  const stateBadge = (w) => {
    if (w.status === 'blocked') {
      return <span className="px-2 py-0.5 rounded text-xs bg-red-900/50 text-red-300">{t('admin.worker_status_blocked')}</span>;
    }
    if (w.status === 'offline') {
      return <span className="px-2 py-0.5 rounded text-xs bg-gray-700 text-gray-400">{t('admin.worker_status_offline')}</span>;
    }
    const workerState = w.resources?.worker_state || 'active';
    const throttle = w.resources?.throttle_level || 'normal';
    const stateMap = {
      active: { bg: 'bg-green-900/50', text: 'text-green-300', dot: 'bg-green-400', label: t('admin.worker_state_active') },
      idle: { bg: 'bg-blue-900/50', text: 'text-blue-300', dot: 'bg-blue-400', label: t('admin.worker_state_idle') },
      resting: { bg: 'bg-yellow-900/50', text: 'text-yellow-300', dot: 'bg-yellow-400', label: t('admin.worker_state_resting') },
    };
    const s = stateMap[workerState] || stateMap.active;
    const showThrottle = throttle === 'warning' || throttle === 'danger';
    return (
      <span className="flex items-center gap-1">
        <span className={`px-2 py-0.5 rounded text-xs ${s.bg} ${s.text} flex items-center gap-1`}>
          <span className={`w-1.5 h-1.5 rounded-full ${s.dot}`} />
          {s.label}
        </span>
        {showThrottle && (
          <AlertOctagon size={12} className={throttle === 'danger' ? 'text-red-400' : 'text-yellow-400'}
            title={t(`admin.worker_throttle_${throttle}`)} />
        )}
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

      {/* Global Processing Mode */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-gray-300">{t('admin.worker_global_mode')}</div>
            <div className="text-xs text-gray-500 mt-0.5">{t('admin.worker_global_mode_desc')}</div>
          </div>
          <div className="flex rounded-lg overflow-hidden border border-gray-600">
            <button
              onClick={() => handleGlobalMode('mc_only')}
              className={`px-4 py-2 text-xs font-medium transition-colors ${
                globalMode === 'mc_only'
                  ? 'bg-amber-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t('admin.worker_mode_mc_only')}
            </button>
            <button
              onClick={() => handleGlobalMode('parse_only')}
              className={`px-4 py-2 text-xs font-medium transition-colors ${
                globalMode === 'parse_only'
                  ? 'bg-teal-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t('admin.worker_mode_parse_only')}
            </button>
            <button
              onClick={() => handleGlobalMode('builtin_worker')}
              className={`px-4 py-2 text-xs font-medium transition-colors ${
                globalMode === 'builtin_worker'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t('admin.worker_mode_builtin')}
            </button>
            <button
              onClick={() => handleGlobalMode('auto')}
              className={`px-4 py-2 text-xs font-medium transition-colors ${
                globalMode === 'auto'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {t('admin.worker_mode_auto')}
            </button>
          </div>
        </div>
        {globalMode === 'mc_only' && (
          <div className="text-xs text-amber-400/70 mt-2">{t('admin.worker_mode_mc_only_desc')}</div>
        )}
        {globalMode === 'parse_only' && (
          <div className="text-xs text-teal-400/70 mt-2">{t('admin.worker_mode_parse_only_desc')}</div>
        )}
        {globalMode === 'builtin_worker' && (
          <div className="text-xs text-purple-400/70 mt-2">{t('admin.worker_mode_builtin_desc')}</div>
        )}
        {globalMode === 'auto' && (
          <div className="text-xs text-blue-400/70 mt-2">{t('admin.worker_mode_auto_desc')}</div>
        )}
      </div>

      {/* Auto Processing (server processes all phases when no workers) — hidden in builtin_worker mode */}
      {globalMode !== 'builtin_worker' && <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-gray-300">{t('worker.auto_title')}</div>
            <div className="text-xs text-gray-500 mt-0.5">{t('worker.auto_desc')}</div>
          </div>
          <button
            onClick={async () => {
              const newVal = !autoProcessing;
              setAutoProcessing(newVal);
              try { await updateAutoProcessing({ enabled: newVal }); } catch (e) { console.error(e); setAutoProcessing(!newVal); }
            }}
            className={`px-4 py-2 text-xs font-medium rounded-lg transition-colors ${
              autoProcessing
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-400 hover:text-white'
            }`}
          >
            {autoProcessing ? t('worker.auto_on') : t('worker.auto_off')}
          </button>
        </div>
        {autoProcessing && (
          <div className="flex items-center gap-3 mt-3 pt-3 border-t border-gray-700/50">
            <div className="text-xs text-gray-400">{t('worker.rest_title')}</div>
            <input
              type="number" min="0" max="300" value={restAfterBatch}
              onChange={async (e) => {
                const v = Math.max(0, Math.min(300, parseInt(e.target.value) || 0));
                setRestAfterBatch(v);
                try { await updateAutoProcessing({ rest_after_batch_s: v }); } catch (err) { console.error(err); }
              }}
              className="w-20 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white"
            />
            <span className="text-xs text-gray-500">{t('worker.rest_unit')}</span>
          </div>
        )}
      </div>}

      {/* Embedded Worker toggle */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-gray-300">{t('worker.embedded_worker')}</div>
            <div className="text-xs text-gray-500 mt-0.5">{t('worker.embedded_desc')}</div>
          </div>
          <div className="flex items-center gap-3">
            {embeddedEnabled && (
              <span className={`text-xs px-2 py-0.5 rounded ${
                embeddedStatus.running
                  ? 'bg-green-900/50 text-green-400'
                  : 'bg-yellow-900/50 text-yellow-400'
              }`}>
                {embeddedStatus.running ? t('worker.status_running') : embeddedStatus.status}
                {embeddedStatus.jobs_completed > 0 && ` · ${embeddedStatus.jobs_completed} ${t('worker.jobs_completed')}`}
              </span>
            )}
            <button
              onClick={async () => {
                const newVal = !embeddedEnabled;
                setEmbeddedEnabled(newVal);
                try {
                  const res = await updateEmbeddedWorker({ enabled: newVal });
                  setEmbeddedStatus({ running: res.running, status: res.status, jobs_completed: res.jobs_completed || 0 });
                } catch (e) {
                  console.error(e);
                  setEmbeddedEnabled(!newVal);
                }
              }}
              className={`px-4 py-2 text-xs font-medium rounded-lg transition-colors ${
                embeddedEnabled
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {embeddedEnabled ? 'ON' : 'OFF'}
            </button>
          </div>
        </div>
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
              <th className="text-left px-4 py-3">{t('admin.worker_state')}</th>
              <th className="text-left px-4 py-3">{t('admin.worker_mode')}</th>
              <th className="text-left px-4 py-3">{t('admin.worker_capacity')}</th>
              <th className="text-left px-4 py-3">{t('admin.worker_speed')}</th>
              <th className="text-left px-4 py-3">{t('admin.worker_resources')}</th>
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
                  <div className="flex items-center gap-1.5">
                    {w.worker_name === '__builtin__' ? (
                      <span className="px-1.5 py-0.5 rounded text-xs font-medium bg-purple-900/50 text-purple-400">
                        {t('admin.worker_builtin_label')}
                      </span>
                    ) : (
                      w.worker_name
                    )}
                  </div>
                  {w.hostname && <div className="text-xs text-gray-500">{w.hostname}</div>}
                </td>
                <td className="px-4 py-3 text-gray-400">{w.worker_name === '__builtin__' ? 'server' : w.username}</td>
                <td className="px-4 py-3">{stateBadge(w)}</td>
                <td className="px-4 py-3">
                  {w.processing_mode_override === 'mc_only' ? (
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-amber-900/50 text-amber-400">
                      MC Only
                    </span>
                  ) : w.processing_mode_override === 'embed_only' ? (
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-orange-900/50 text-orange-400"
                      title={t('admin.worker_capability_tooltip')}>
                      {t('admin.worker_capability_lightweight')}
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-900/50 text-green-400"
                      title={t('admin.worker_capability_tooltip')}>
                      {t('admin.worker_capability_full')}
                    </span>
                  )}
                </td>
                <td className="px-4 py-3">
                  {editingCapacity?.id === w.id ? (
                    <input
                      type="number"
                      min={1}
                      max={32}
                      value={editingCapacity.value}
                      onChange={(e) => setEditingCapacity({ id: w.id, value: Math.max(1, Math.min(32, Number(e.target.value))) })}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleCapacitySave(w.id);
                        if (e.key === 'Escape') setEditingCapacity(null);
                      }}
                      onBlur={() => handleCapacitySave(w.id)}
                      autoFocus
                      className="w-14 bg-gray-900 border border-blue-500 rounded px-2 py-0.5 text-xs font-mono text-yellow-300 focus:outline-none"
                    />
                  ) : (
                    <span
                      className="font-mono text-yellow-300 cursor-pointer hover:text-yellow-200 flex items-center gap-1 group"
                      onClick={() => setEditingCapacity({ id: w.id, value: w.batch_capacity_override || w.batch_capacity })}
                      title={t('admin.worker_edit_capacity')}
                    >
                      B:{w.batch_capacity_override || w.batch_capacity}
                      <Pencil size={10} className="text-gray-600 group-hover:text-yellow-300" />
                    </span>
                  )}
                </td>
                <td className="px-4 py-3">
                  {w.throughput > 0 ? (
                    <span className="font-mono text-emerald-400">{w.throughput.toFixed(1)}<span className="text-emerald-400/50 text-xs ml-0.5">/m</span></span>
                  ) : (
                    <span className="text-gray-600">-</span>
                  )}
                </td>
                <td className="px-4 py-3">
                  <ResourceMetrics resources={w.resources} t={t} />
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

// ── Members Panel ────────────────────────────────────────

function MembersPanel({ currentUser }) {
  const { t } = useLocale();
  const [members, setMembers] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await listMembers();
      setMembers(data.members || []);
    } catch (e) {
      console.error('Failed to load members:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const isSelf = (m) => {
    if (!currentUser) return false;
    return m.id === currentUser.uid || m.id === currentUser.id || m.email === currentUser.email;
  };

  const handleRoleChange = async (m, newRole) => {
    if (isSelf(m)) return;
    try {
      await updateMemberRole(m.id, newRole);
      load();
    } catch (e) {
      console.error('Failed to update member role:', e);
    }
  };

  const handleToggleActive = async (m) => {
    if (isSelf(m)) return;
    try {
      if (m.is_active) {
        await deactivateMember(m.id);
      } else {
        await activateMember(m.id);
      }
      load();
    } catch (e) {
      console.error('Failed to toggle member status:', e);
    }
  };

  const handleRemove = async (m) => {
    if (isSelf(m)) return;
    if (!confirm(t('members.confirm_remove'))) return;
    try {
      await removeMember(m.id);
      load();
    } catch (e) {
      console.error('Failed to remove member:', e);
    }
  };

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  return (
    <div>
      <h2 className="text-lg font-semibold mb-4">{t('members.title')}</h2>
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400">
              <th className="text-left px-4 py-3">{t('members.col_email')}</th>
              <th className="text-left px-4 py-3">{t('members.col_display_name')}</th>
              <th className="text-left px-4 py-3">{t('members.col_role')}</th>
              <th className="text-left px-4 py-3">{t('members.col_status')}</th>
              <th className="text-left px-4 py-3">{t('members.col_joined')}</th>
              <th className="text-right px-4 py-3">{t('members.col_actions')}</th>
            </tr>
          </thead>
          <tbody>
            {members.map((m) => (
              <tr key={m.id} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                <td className="px-4 py-3 font-medium">
                  {m.email}
                  {isSelf(m) && (
                    <span className="ml-2 text-xs text-blue-400">({t('members.cannot_change_self')})</span>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-400">{m.display_name || '-'}</td>
                <td className="px-4 py-3">
                  {isSelf(m) ? (
                    <span className={`px-2 py-0.5 rounded text-xs ${
                      m.role === 'admin' ? 'bg-purple-900/50 text-purple-300' : 'bg-gray-700 text-gray-300'
                    }`}>
                      {m.role}
                    </span>
                  ) : (
                    <select
                      value={m.role || 'user'}
                      onChange={(e) => handleRoleChange(m, e.target.value)}
                      className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white focus:outline-none focus:border-blue-500"
                    >
                      <option value="admin">{t('members.role_admin')}</option>
                      <option value="user">{t('members.role_user')}</option>
                    </select>
                  )}
                </td>
                <td className="px-4 py-3">
                  {m.is_active !== false ? (
                    <span className="flex items-center gap-1 text-green-400 text-xs">
                      <CheckCircle size={14} />
                      {t('members.status_active')}
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-red-400 text-xs">
                      <XCircle size={14} />
                      {t('members.status_inactive')}
                    </span>
                  )}
                </td>
                <td className="px-4 py-3 text-gray-500 text-xs">
                  {m.joined_at ? new Date(m.joined_at).toLocaleDateString() : m.created_at ? new Date(m.created_at).toLocaleDateString() : '-'}
                </td>
                <td className="px-4 py-3 text-right">
                  {!isSelf(m) && (
                    <div className="flex gap-1 justify-end">
                      <button
                        onClick={() => handleToggleActive(m)}
                        className="p-1.5 rounded hover:bg-gray-600 text-gray-400 hover:text-white"
                        title={m.is_active !== false ? t('admin.action_deactivate') : t('admin.action_activate')}
                      >
                        {m.is_active !== false ? <ShieldOff size={14} /> : <Shield size={14} />}
                      </button>
                      <button
                        onClick={() => handleRemove(m)}
                        className="p-1.5 rounded hover:bg-red-900/50 text-gray-400 hover:text-red-400"
                        title={t('admin.action_delete_user')}
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {members.length === 0 && (
          <div className="text-center text-gray-500 py-8 text-sm">{t('members.no_members')}</div>
        )}
      </div>
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

// ── Thumbnails Panel ────────────────────────────────────

function ThumbnailsPanel() {
  const { t } = useLocale();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const result = await getThumbnailStats();
      if (result?.success) setData(result);
    } catch (e) {
      console.error('Failed to fetch thumbnail stats:', e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  if (loading && !data) {
    return <div className="flex items-center gap-2 text-gray-400"><Loader2 className="animate-spin" size={16} /> Loading...</div>;
  }
  if (!data || !data.sources?.length) {
    return <div className="text-gray-500">{t('admin.thumb_no_data')}</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{t('admin.thumb_title')}</h2>
        <button onClick={fetchData} className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors">
          <RefreshCw size={16} />
        </button>
      </div>

      {/* Grand total */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-400">{t('admin.thumb_total')}</span>
          <span className="font-mono font-bold text-white">{data.grand_total.toLocaleString()}</span>
        </div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-green-400">{t('admin.thumb_has_thumb')}</span>
          <span className="font-mono font-bold text-green-400">{data.grand_has_thumb.toLocaleString()}</span>
        </div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm text-amber-400">{t('admin.thumb_missing')}</span>
          <span className="font-mono font-bold text-amber-400">{data.grand_missing.toLocaleString()}</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden flex">
          <div className="bg-green-500 h-full transition-all" style={{ width: `${data.grand_total > 0 ? (data.grand_has_thumb / data.grand_total * 100) : 0}%` }} />
          <div className="bg-amber-500 h-full transition-all" style={{ width: `${data.grand_total > 0 ? (data.grand_missing / data.grand_total * 100) : 0}%` }} />
        </div>
        <div className="text-right text-[10px] text-gray-500 mt-1">
          {data.grand_total > 0 ? Math.round(data.grand_has_thumb / data.grand_total * 100) : 0}%
        </div>
      </div>

      {/* Per-source breakdown */}
      <div className="space-y-2">
        {data.sources.map(src => {
          const pct = src.total > 0 ? Math.round(src.has_thumb / src.total * 100) : 0;
          return (
            <div key={src.source} className="bg-gray-800/50 rounded-lg p-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-sm font-medium text-gray-300 truncate max-w-[300px]" title={src.source}>
                  {src.source.startsWith('webdav://') ? src.source.replace('webdav://', '📁 ') : src.source}
                </span>
                <span className="text-xs text-gray-500">{src.total.toLocaleString()} files</span>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <span className="text-green-400">{src.has_thumb}</span>
                <span className="text-gray-600">/</span>
                <span className="text-amber-400">{src.missing} {t('admin.thumb_missing')}</span>
                <div className="flex-1 bg-gray-700 rounded-full h-1.5 overflow-hidden flex">
                  <div className="bg-green-500 h-full" style={{ width: `${pct}%` }} />
                </div>
                <span className="font-mono text-gray-400 text-[10px]">{pct}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


// ── Queue Panel ──────────────────────────────────────────

function QueuePanel() {
  const { t } = useLocale();
  const [stats, setStats] = useState(null);
  const [thumbStats, setThumbStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [cleanupMsg, setCleanupMsg] = useState('');
  const [workRequests, setWorkRequests] = useState([]);
  const [showCompleted, setShowCompleted] = useState(false);
  const [expandedWR, setExpandedWR] = useState(new Set());
  const [wrDetails, setWrDetails] = useState({});

  const load = useCallback(async () => {
    try {
      const [jobData, thumbData, wrData] = await Promise.all([
        getJobStats(),
        getThumbnailStats().catch(() => null),
        getWorkRequests(showCompleted).catch(() => []),
      ]);
      setStats(jobData);
      if (thumbData && thumbData.success !== false) {
        setThumbStats(thumbData);
      }
      setWorkRequests(Array.isArray(wrData) ? wrData : (wrData?.work_requests || []));
    } catch (e) {
      console.error('Failed to load queue stats:', e);
    }
    setLoading(false);
  }, [showCompleted]);

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

  const handleQueueCleanup = async () => {
    try {
      const data = await cleanupQueue();
      setCleanupMsg(t('admin.queue_cleanup_result', { count: data.total_removed || 0 }));
      load();
      setTimeout(() => setCleanupMsg(''), 5000);
    } catch (e) {
      console.error('Queue cleanup failed:', e);
    }
  };

  const handleRecoveryScan = async () => {
    try {
      setCleanupMsg('Scanning...');
      const data = await runRecoveryScan();
      const msg = data.repaired_files > 0
        ? `Recovery: ${data.repaired_files} files → ${data.recovery_wrs_created || 0} WR(s)`
        : 'Recovery: all files complete';
      setCleanupMsg(msg);
      load();
      setTimeout(() => setCleanupMsg(''), 8000);
    } catch (e) {
      console.error('Recovery scan failed:', e);
      setCleanupMsg('Recovery scan failed');
      setTimeout(() => setCleanupMsg(''), 5000);
    }
  };

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  const bufferReady = (stats?.parse_ahead_parsed || 0);
  const statItems = [
    ...(stats?.download_waiting > 0
      ? [{ key: 'download_waiting', label: t('admin.queue_download_waiting'), color: 'bg-cyan-600' }]
      : []),
    { key: 'pending', label: t('admin.queue_pending'), color: 'bg-yellow-500' },
    ...(bufferReady > 0 ? [{ key: 'parse_ahead_parsed', label: t('admin.queue_buffer'), color: 'bg-teal-500' }] : []),
    { key: 'assigned', label: t('admin.queue_assigned'), color: 'bg-blue-500' },
    { key: 'processing', label: t('admin.queue_processing'), color: 'bg-cyan-500' },
    { key: 'completed', label: t('admin.queue_completed'), color: 'bg-green-500' },
    { key: 'failed', label: t('admin.queue_failed'), color: 'bg-red-500' },
  ];

  const throughput = stats?.throughput ?? 0;
  // Exclude download_waiting from remaining (can't be processed until downloaded)
  const remaining = (stats?.pending ?? 0) - (stats?.download_waiting ?? 0)
                  + (stats?.assigned ?? 0) + (stats?.processing ?? 0);
  const etaMin = throughput > 0 ? Math.ceil(remaining / throughput) : null;

  const toggleExpand = async (wrId) => {
    setExpandedWR(prev => {
      const next = new Set(prev);
      next.has(wrId) ? next.delete(wrId) : next.add(wrId);
      return next;
    });
    // Fetch detail (subtasks) if not yet loaded
    if (!wrDetails[wrId]) {
      try {
        const detail = await getWorkRequestDetail(wrId);
        setWrDetails(prev => ({ ...prev, [wrId]: detail }));
      } catch (e) {
        console.error('Failed to load WR detail:', e);
      }
    }
  };

  const handlePauseResume = async (wr) => {
    try {
      if (wr.status === 'paused') {
        await resumeWorkRequest(wr.id);
      } else {
        await pauseWorkRequest(wr.id);
      }
      load();
    } catch (e) {
      console.error('Pause/resume failed:', e);
    }
  };

  const handleCancelWR = async (wrId) => {
    try {
      await cancelWorkRequest(wrId);
      load();
    } catch (e) {
      console.error('Cancel WR failed:', e);
    }
  };

  const wrStatusColor = (status) => {
    switch (status) {
      case 'queued': return 'text-yellow-400';
      case 'processing': return 'text-blue-400';
      case 'completed': return 'text-green-400';
      case 'paused': return 'text-orange-400';
      case 'cancelled': return 'text-gray-500';
      default: return 'text-gray-400';
    }
  };

  const wrStatusLabel = (status) => {
    const key = `admin.wr_status_${status}`;
    return t(key);
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">{t('admin.queue_title')}</h2>
        <div className="flex items-center gap-2">
          {cleanupMsg && <span className="text-xs text-green-400">{cleanupMsg}</span>}
          <button
            onClick={handleRecoveryScan}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-amber-700 hover:bg-amber-600 rounded text-xs text-gray-200"
          >
            <AlertTriangle size={12} />
            Recovery Scan
          </button>
          <button
            onClick={handleQueueCleanup}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs text-gray-300"
          >
            <Trash2 size={12} />
            {t('admin.queue_cleanup')}
          </button>
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

      {/* Work Requests */}
      {workRequests.length > 0 && (
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300">{t('admin.wr_title')}</h3>
            <label className="flex items-center gap-1.5 text-xs text-gray-500 cursor-pointer">
              <input
                type="checkbox"
                checked={showCompleted}
                onChange={(e) => setShowCompleted(e.target.checked)}
                className="rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-0 w-3 h-3"
              />
              {t('admin.wr_show_completed')}
            </label>
          </div>
          <div className="space-y-2">
            {workRequests.map((wr, idx) => {
              const pct = wr.total_files > 0 ? ((wr.completed_count / wr.total_files) * 100) : 0;
              const isExpanded = expandedWR.has(wr.id);
              const isDone = wr.status === 'completed' || wr.status === 'cancelled';
              return (
                <div key={wr.id}>
                  <div
                    className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                      isDone ? 'bg-gray-900/50' : 'bg-gray-900 hover:bg-gray-850'
                    }`}
                    onClick={() => (wrDetails[wr.id]?.subtasks?.length > 0 || !wrDetails[wr.id]) && toggleExpand(wr.id)}
                  >
                    {/* Order number */}
                    <span className="text-xs text-gray-600 font-mono w-5 flex-shrink-0">{idx + 1}</span>
                    {/* Expand arrow */}
                    {wrDetails[wr.id]?.subtasks?.length > 0 ? (
                      <ChevronRight size={14} className={`text-gray-500 transition-transform flex-shrink-0 ${isExpanded ? 'rotate-90' : ''}`} />
                    ) : (
                      <span className="w-3.5 flex-shrink-0" />
                    )}
                    {/* Name */}
                    {wr.name?.startsWith('[Recovery]') && (
                      <span className="text-[10px] px-1 py-0.5 rounded bg-amber-800 text-amber-200 flex-shrink-0">
                        Recovery
                      </span>
                    )}
                    <span className={`text-sm flex-shrink-0 max-w-[180px] truncate ${isDone ? 'text-gray-500' : 'text-gray-200'}`}>
                      {wr.name?.startsWith('[Recovery] ') ? wr.name.slice(11) : wr.name}
                    </span>
                    {/* Status badge */}
                    <span className={`text-[10px] px-1.5 py-0.5 rounded flex-shrink-0 ${wrStatusColor(wr.status)} bg-gray-800`}>
                      {wrStatusLabel(wr.status)}
                    </span>
                    {/* Progress bar */}
                    <div className="flex-1 flex items-center gap-2 min-w-0">
                      <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            wr.status === 'completed' ? 'bg-green-500' :
                            wr.status === 'paused' ? 'bg-orange-500' :
                            wr.status === 'cancelled' ? 'bg-gray-600' : 'bg-blue-500'
                          }`}
                          style={{ width: `${Math.min(pct, 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 font-mono flex-shrink-0 w-20 text-right">
                        {t('admin.wr_progress', { completed: wr.completed_count, total: wr.total_files })}
                      </span>
                    </div>
                    {/* Failed count */}
                    {wr.failed_count > 0 && (
                      <span className="text-[10px] text-red-400 flex-shrink-0">{wr.failed_count} failed</span>
                    )}
                    {/* Actions */}
                    {!isDone && (
                      <div className="flex items-center gap-1 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
                        <button
                          onClick={() => handlePauseResume(wr)}
                          className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white"
                          title={wr.status === 'paused' ? t('admin.wr_resume') : t('admin.wr_pause')}
                        >
                          {wr.status === 'paused' ? <Play size={12} /> : <Pause size={12} />}
                        </button>
                        <button
                          onClick={() => handleCancelWR(wr.id)}
                          className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-red-400"
                          title={t('admin.wr_cancel')}
                        >
                          <CircleX size={12} />
                        </button>
                      </div>
                    )}
                  </div>
                  {/* Sub-tasks */}
                  {isExpanded && wrDetails[wr.id]?.subtasks?.length > 0 && (
                    <div className="ml-12 mt-1 space-y-1">
                      {wrDetails[wr.id].subtasks.map((st) => {
                        const stPct = st.total_files > 0 ? ((st.completed_count / st.total_files) * 100) : 0;
                        const stDone = st.completed_count + st.failed_count >= st.total_files;
                        return (
                          <div key={st.id} className="flex items-center gap-2 px-2 py-1 text-xs">
                            <Folder size={12} className={stDone ? 'text-green-500' : 'text-gray-500'} />
                            <span className={`w-28 truncate ${stDone ? 'text-gray-500' : 'text-gray-300'}`}>
                              {st.folder_name}
                            </span>
                            <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${stDone ? 'bg-green-500' : 'bg-blue-500'}`}
                                style={{ width: `${Math.min(stPct, 100)}%` }}
                              />
                            </div>
                            <span className="text-gray-500 font-mono w-16 text-right">
                              {st.completed_count}/{st.total_files}
                            </span>
                            {st.failed_count > 0 && (
                              <span className="text-red-400">{st.failed_count}F</span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

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

      {/* Pipeline Progress — cumulative stage bars */}
      {stats && stats.total_files > 0 && (() => {
        const total = stats.total_files;
        const complete = stats.complete_files || 0;
        const dlWaiting = stats.download_waiting || 0;
        const dlDone = total - dlWaiting;
        // parseDone = files actually parsed (exclude download-waiting)
        const parseDone = (stats.phase_parse_done || 0) + complete;
        const mcDone = (stats.phase_vision_done || 0) + complete;
        const embedDone = (stats.phase_embed_done || 0) + complete;

        // Build stage list — only show download bar when WebDAV files exist
        const stages = [
          ...(dlWaiting > 0 ? [{
            label: t('admin.queue_stage_download'),
            done: dlDone, total, color: 'bg-cyan-500', textColor: 'text-cyan-400',
            detail: stats.download_buffer
              ? `${t('admin.queue_download_in_flight')}: ${stats.download_buffer.in_flight} | ${t('admin.queue_download_buffered')}: ${stats.download_buffer.active_files}/${stats.download_buffer.max_files}`
              : null,
          }] : []),
          { label: t('admin.queue_stage_parse'), done: parseDone, total, color: 'bg-teal-500', textColor: 'text-teal-400',
            detail: (stats.parse_ahead_parsing || 0) > 0 ? `${t('admin.queue_parse_ahead_parsing')}: ${stats.parse_ahead_parsing}` : null },
          { label: t('admin.queue_stage_mc'), done: mcDone, total, color: 'bg-purple-500', textColor: 'text-purple-400' },
          { label: t('admin.queue_stage_embed'), done: embedDone, total, color: 'bg-blue-500', textColor: 'text-blue-400' },
          { label: t('admin.queue_stage_complete'), done: complete, total, color: 'bg-green-500', textColor: 'text-green-400' },
        ];

        return (
          <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
            <div className="text-xs text-gray-400 mb-3 font-medium">
              {t('admin.queue_pipeline_progress')}
              <span className="ml-2 text-gray-500">
                {complete < total
                  ? `(${total - complete} ${t('admin.queue_phase_remaining')})`
                  : `(${t('admin.queue_phase_all_done')})`}
              </span>
            </div>
            <div className="space-y-2">
              {stages.map(({ label, done, total: stageTotal, color, textColor, detail }) => {
                const pct = stageTotal > 0 ? (done / stageTotal) * 100 : 0;
                return (
                  <div key={label}>
                    <div className="flex items-center gap-3">
                      <span className={`text-xs w-20 flex-shrink-0 ${textColor}`}>{label}</span>
                      <div className="flex-1 h-2 bg-gray-900 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${color}`}
                          style={{ width: `${Math.min(pct, 100)}%` }}
                        />
                      </div>
                      <span className="text-xs text-gray-400 font-mono w-24 text-right flex-shrink-0">
                        {done}/{stageTotal}
                      </span>
                    </div>
                    {detail && (
                      <div className="ml-[calc(5rem+0.75rem)] text-[10px] text-gray-500 mt-0.5">{detail}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })()}

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


// ── Classification Panel ─────────────────────────────────

function ClassificationPanel() {
  const { t } = useLocale();
  const [domains, setDomains] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [expandedTypes, setExpandedTypes] = useState(new Set());
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);

  const load = useCallback(async () => {
    try {
      const [domainList, activeConfig] = await Promise.all([
        listDomains(),
        getActiveDomainConfig(),
      ]);
      setDomains(domainList || []);
      const currentId = activeConfig?.active_domain || null;
      setActiveId(currentId);
      // Auto-select active domain or first
      const autoSelect = currentId || (domainList?.[0]?.id || null);
      if (autoSelect) {
        const detail = await getDomainDetail(autoSelect);
        setSelectedDomain(detail);
      }
    } catch (e) {
      console.error('Failed to load classification domains:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleActivate = async (domainId) => {
    if (saving || domainId === activeId) return;
    setSaving(true);
    try {
      await setActiveDomain(domainId);
      setActiveId(domainId);
    } catch (e) {
      console.error('Failed to activate domain:', e);
    }
    setSaving(false);
  };

  const handleSelectDomain = async (domainId) => {
    if (selectedDomain?.id === domainId) return;
    try {
      const detail = await getDomainDetail(domainId);
      setSelectedDomain(detail);
      setExpandedTypes(new Set());
    } catch (e) {
      console.error('Failed to load domain detail:', e);
    }
  };

  const toggleType = (type) => {
    setExpandedTypes(prev => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-semibold">{t('admin.classification_title')}</h2>
          <p className="text-sm text-gray-400 mt-1">{t('admin.classification_desc')}</p>
        </div>
        <button
          onClick={() => setShowCreateModal(true)}
          className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm text-white transition-colors flex-shrink-0"
        >
          <Plus size={14} />
          {t('admin.classification_create')}
        </button>
      </div>

      {/* Domain Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
        {domains.map(d => {
          const isActive = d.id === activeId;
          const isSelected = d.id === selectedDomain?.id;
          return (
            <div
              key={d.id}
              onClick={() => handleSelectDomain(d.id)}
              className={`bg-gray-800 rounded-lg border p-4 cursor-pointer transition-colors ${
                isSelected
                  ? isActive ? 'border-green-500 ring-1 ring-green-500/30' : 'border-blue-500 ring-1 ring-blue-500/30'
                  : isActive ? 'border-green-500/50' : 'border-gray-700 hover:border-gray-500'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium truncate">{d.name}</h3>
                {isActive ? (
                  <span className="bg-green-900/50 text-green-300 rounded px-2 py-0.5 text-xs flex-shrink-0">
                    {t('admin.classification_active')}
                  </span>
                ) : (
                  <button
                    onClick={(e) => { e.stopPropagation(); handleActivate(d.id); }}
                    disabled={saving}
                    className="bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded text-xs px-3 py-1 flex-shrink-0 transition-colors"
                  >
                    {saving ? '...' : t('admin.classification_activate')}
                  </button>
                )}
              </div>
              <p className="text-xs text-gray-400 mb-2 line-clamp-2">{d.description}</p>
              <div className="text-xs text-gray-500">
                {t('admin.classification_types_count', { count: d.image_types_count || d.image_types?.length || 0 })}
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Domain Detail */}
      {selectedDomain ? (
        <div className="space-y-4">
          {/* Image Types */}
          <div>
            <h3 className="text-sm font-medium text-gray-300 mb-2">{t('admin.classification_image_types')}</h3>
            <div className="flex flex-wrap gap-2">
              {(selectedDomain.image_types || []).map(type => (
                <span
                  key={type}
                  className="bg-blue-900/50 text-blue-300 rounded px-2.5 py-1 text-xs"
                >
                  {type}
                </span>
              ))}
            </div>
          </div>

          {/* Type Hints Accordion */}
          <div>
            <h3 className="text-sm font-medium text-gray-300 mb-2">{t('admin.classification_type_hints')}</h3>
            <div className="bg-gray-800 rounded-lg border border-gray-700 divide-y divide-gray-700">
              {Object.entries(selectedDomain.type_hints || {}).map(([type, hints]) => (
                <div key={type}>
                  <button
                    onClick={() => toggleType(type)}
                    className="w-full flex items-center gap-2 px-4 py-3 text-sm hover:bg-gray-700/30 transition-colors"
                  >
                    <ChevronDown
                      size={14}
                      className={`text-gray-400 transition-transform ${expandedTypes.has(type) ? '' : '-rotate-90'}`}
                    />
                    <span className="font-medium text-gray-200">{type}</span>
                    <span className="text-xs text-gray-500 ml-auto">
                      {Object.keys(hints).length} hints
                    </span>
                  </button>
                  {expandedTypes.has(type) && (
                    <div className="px-4 pb-3 space-y-1.5">
                      {Object.entries(hints).length > 0 ? (
                        Object.entries(hints).map(([key, values]) => (
                          <div key={key} className="flex gap-2 text-xs">
                            <span className="text-gray-400 min-w-[120px] flex-shrink-0">{key}:</span>
                            <span className="text-gray-300">
                              {Array.isArray(values) ? values.join(', ') : String(values)}
                            </span>
                          </div>
                        ))
                      ) : (
                        <div className="text-xs text-gray-500">{t('admin.classification_no_hints')}</div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Common Hints */}
          {selectedDomain.common_hints && Object.keys(selectedDomain.common_hints).length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-300 mb-2">{t('admin.classification_common_hints')}</h3>
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-1.5">
                {Object.entries(selectedDomain.common_hints).map(([key, values]) => (
                  <div key={key} className="flex gap-2 text-xs">
                    <span className="text-gray-400 min-w-[120px] flex-shrink-0">{key}:</span>
                    <span className="text-gray-300">
                      {Array.isArray(values) ? values.join(', ') : String(values)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Type Instructions */}
          {selectedDomain.type_instructions && Object.keys(selectedDomain.type_instructions).length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-gray-300 mb-2">{t('admin.classification_type_instructions')}</h3>
              <div className="bg-gray-800 rounded-lg border border-gray-700 divide-y divide-gray-700">
                {Object.entries(selectedDomain.type_instructions).map(([type, instruction]) => (
                  <div key={type} className="px-4 py-3">
                    <span className="text-xs font-medium text-gray-200">{type}</span>
                    <p className="text-xs text-gray-400 mt-1">{instruction}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-sm text-gray-500">{t('admin.classification_no_domain')}</div>
      )}

      {/* Create Domain Modal */}
      {showCreateModal && (
        <CreateDomainModal
          existingIds={domains.map(d => d.id)}
          onClose={() => setShowCreateModal(false)}
          onCreated={() => { setShowCreateModal(false); load(); }}
        />
      )}
    </div>
  );
}


// ── Prompt Generator ─────────────────────────────────────

function generateDomainPrompt({ domainId, nameEn, nameKo, description }) {
  return `You are a domain profile generator for ImageParser, an AI-powered image classification system.

Create a YAML domain profile for: "${nameEn}" (${nameKo})
Description/Use case: ${description}

## YAML Schema (MUST follow exactly)

domain:
  id: ${domainId}
  name: "${nameEn}"
  name_ko: "${nameKo}"
  description: "${description}"

image_types:
  - type1
  - type2
  # ... (at least 3, at most 12 types)

type_hints:
  type1:
    field_name:
      - value1
      - value2
    another_field:
      - value1
  type2:
    field_name:
      - value1
  # Each image_type MUST have at least one hint field
  # Each hint field MUST be a list of 3-15 string values
  # Values must be snake_case

type_instructions:
  type1: "Extra instruction text for AI when analyzing this type"
  # Optional per type, but recommended for at least 3 types

## Example (game_asset domain, abbreviated)

domain:
  id: game_asset
  name: "Game Asset"
  name_ko: "게임 에셋"
  description: "Game development assets including characters, backgrounds, UI elements, items, and effects."

image_types:
  - character
  - background
  - ui_element
  - item
  - icon
  - texture
  - effect

type_hints:
  character:
    character_type:
      - hero
      - villain
      - NPC
      - boss
      - monster
    pose:
      - standing
      - action
      - portrait
      - full_body
    emotion:
      - neutral
      - happy
      - angry
      - sad
      - determined
  background:
    scene_type:
      - dungeon
      - castle
      - forest
      - village
      - arena
    time_of_day:
      - dawn
      - morning
      - noon
      - sunset
      - night
  item:
    item_type:
      - weapon
      - armor
      - potion
      - accessory
      - material
    rarity_feel:
      - common
      - uncommon
      - rare
      - epic
      - legendary

type_instructions:
  character: "Identify character class/role from visual cues (armor type, weapons, accessories). Note if this is a sprite sheet or single illustration."
  background: "Pay special attention to game-specific scene elements like interactive objects, spawn points, and pathways."
  item: "Identify item rarity from visual cues (glow effects, color coding, particle effects)."

## Rules
1. Output ONLY the YAML content — no markdown fences, no explanations, no commentary
2. image_types: 3-12 items, lowercase snake_case
3. type_hints: every image_type MUST have at least 1 hint field with 3-15 values
4. Hint field names and values: snake_case
5. type_instructions: provide for at least the top 3 most complex types
6. Do NOT include common_hints (art_style, color_palette) — those come from _base.yaml automatically
7. domain.id MUST be exactly: ${domainId}
8. Make the hints genuinely useful for classifying "${nameEn}" images`;
}


// ── Create Domain Modal ──────────────────────────────────

function CreateDomainModal({ existingIds, onClose, onCreated }) {
  const { t } = useLocale();
  const [step, setStep] = useState(1);
  const [domainId, setDomainId] = useState('');
  const [nameEn, setNameEn] = useState('');
  const [nameKo, setNameKo] = useState('');
  const [description, setDescription] = useState('');
  const [generatedPrompt, setGeneratedPrompt] = useState('');
  const [yamlInput, setYamlInput] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const idError = domainId && !/^[a-z][a-z0-9_]*$/.test(domainId)
    ? t('admin.classification_id_invalid')
    : domainId && existingIds.includes(domainId)
      ? t('admin.classification_id_exists')
      : '';

  const canNext1 = domainId && nameEn && nameKo && description && !idError;

  const handleNext1 = () => {
    const prompt = generateDomainPrompt({ domainId, nameEn, nameKo, description });
    setGeneratedPrompt(prompt);
    setStep(2);
  };

  const handleCopyPrompt = async () => {
    try {
      await navigator.clipboard.writeText(generatedPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for non-HTTPS contexts
      const textarea = document.createElement('textarea');
      textarea.value = generatedPrompt;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleSave = async () => {
    setError('');
    setSaving(true);
    try {
      // Strip markdown code fences if present
      const cleaned = yamlInput
        .replace(/^```[\w]*\n?/, '')
        .replace(/\n?```\s*$/, '')
        .trim();

      const result = await saveDomainYaml(domainId, cleaned);
      if (result?.success) {
        onCreated();
      } else {
        setError(result?.error || result?.detail || t('admin.classification_save_error'));
      }
    } catch (e) {
      setError(e?.detail || e?.message || t('admin.classification_save_error'));
    }
    setSaving(false);
  };

  const stepTitles = [
    t('admin.classification_step1_title'),
    t('admin.classification_step2_title'),
    t('admin.classification_step3_title'),
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={onClose}>
      <div
        className="bg-gray-800 border border-gray-600 rounded-xl w-full max-w-2xl max-h-[80vh] overflow-y-auto p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Step Indicator */}
        <div className="flex items-center gap-3 mb-6">
          {[1, 2, 3].map(s => (
            <div key={s} className="flex items-center gap-2">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                s === step ? 'bg-blue-600 text-white' : s < step ? 'bg-green-600 text-white' : 'bg-gray-700 text-gray-400'
              }`}>{s}</div>
              {s < 3 && <div className={`w-8 h-0.5 ${s < step ? 'bg-green-600' : 'bg-gray-700'}`} />}
            </div>
          ))}
          <span className="text-sm text-gray-300 ml-2">{stepTitles[step - 1]}</span>
        </div>

        {/* Step 1: Define Domain */}
        {step === 1 && (
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('admin.classification_domain_id')}</label>
              <input
                type="text"
                value={domainId}
                onChange={(e) => setDomainId(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))}
                placeholder={t('admin.classification_domain_id_placeholder')}
                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
              />
              {idError && <p className="text-xs text-red-400 mt-1">{idError}</p>}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('admin.classification_domain_name_en')}</label>
                <input
                  type="text"
                  value={nameEn}
                  onChange={(e) => setNameEn(e.target.value)}
                  placeholder="e.g. Medical Image"
                  className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">{t('admin.classification_domain_name_ko')}</label>
                <input
                  type="text"
                  value={nameKo}
                  onChange={(e) => setNameKo(e.target.value)}
                  placeholder="예: 의료 이미지"
                  className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none"
                />
              </div>
            </div>
            <div>
              <label className="block text-xs text-gray-400 mb-1">{t('admin.classification_domain_use_case')}</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder={t('admin.classification_domain_use_case_placeholder')}
                rows={4}
                className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-sm text-white placeholder-gray-500 focus:border-blue-500 focus:outline-none resize-none"
              />
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <button onClick={onClose} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors">
                {t('admin.classification_cancel')}
              </button>
              <button
                onClick={handleNext1}
                disabled={!canNext1}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded text-sm text-white transition-colors"
              >
                {t('admin.classification_next')}
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Copy Prompt */}
        {step === 2 && (
          <div className="space-y-4">
            <p className="text-sm text-gray-300">{t('admin.classification_prompt_generated')}</p>
            <pre className="bg-gray-900 border border-gray-600 rounded p-3 text-xs text-gray-300 max-h-[40vh] overflow-y-auto whitespace-pre-wrap font-mono">
              {generatedPrompt}
            </pre>
            <div className="flex justify-between items-center pt-2">
              <button onClick={() => setStep(1)} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors">
                {t('admin.classification_back')}
              </button>
              <div className="flex gap-2">
                <button
                  onClick={handleCopyPrompt}
                  className="flex items-center gap-1.5 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 rounded text-sm text-white transition-colors"
                >
                  {copied ? <CheckCircle size={14} /> : <Copy size={14} />}
                  {copied ? t('admin.classification_prompt_copied') : t('admin.classification_copy_prompt')}
                </button>
                <button
                  onClick={() => setStep(3)}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm text-white transition-colors"
                >
                  {t('admin.classification_next')}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 3: Paste YAML */}
        {step === 3 && (
          <div className="space-y-4">
            <label className="block text-xs text-gray-400">{t('admin.classification_paste_yaml')}</label>
            <textarea
              value={yamlInput}
              onChange={(e) => { setYamlInput(e.target.value); setError(''); }}
              rows={16}
              placeholder="domain:\n  id: ..."
              className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-xs text-gray-200 placeholder-gray-600 focus:border-blue-500 focus:outline-none resize-none font-mono"
            />
            {error && (
              <div className="flex items-start gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800/40 rounded p-3">
                <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
            <div className="flex justify-between items-center pt-2">
              <button onClick={() => setStep(2)} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition-colors">
                {t('admin.classification_back')}
              </button>
              <button
                onClick={handleSave}
                disabled={saving || !yamlInput.trim()}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed rounded text-sm text-white transition-colors"
              >
                {saving ? t('admin.classification_saving') : t('admin.classification_save_domain')}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
