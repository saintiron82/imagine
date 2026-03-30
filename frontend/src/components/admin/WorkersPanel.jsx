/**
 * WorkersPanel — worker session management + auto-processing config.
 * Extracted from AdminPage.jsx.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import {
  listWorkerSessions, stopWorkerSession, blockWorkerSession,
  updateWorkerConfig,
  getAutoProcessing, updateAutoProcessing,
  getEmbeddedWorker,
} from '../../api/admin';
import {
  RefreshCw, Square, Ban, Pencil, AlertOctagon, Loader2,
} from 'lucide-react';


// ── Resource Metrics Display ─────────────────────────────

/**
 * Compact resource metrics cell for the workers table.
 * Shows GPU memory bar + temperature, CPU %, and RAM % in a stacked layout.
 * Gracefully handles null/missing data.
 */
export function ResourceMetrics({ resources, t }) {
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

export default function WorkersPanel() {
  const { t } = useLocale();
  const [workers, setWorkers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [editingCapacity, setEditingCapacity] = useState(null); // { id, value }
  const [autoProcessing, setAutoProcessing] = useState(true);
  const [restAfterBatch, setRestAfterBatch] = useState(30);
  const [batchSize, setBatchSize] = useState(5);
  const [verboseLog, setVerboseLog] = useState(false);
  const [embeddedEnabled, setEmbeddedEnabled] = useState(false);
  const [embeddedStatus, setEmbeddedStatus] = useState({ running: false, status: 'idle', jobs_completed: 0 });
  const [dbStats, setDbStats] = useState(null);

  const load = useCallback(async () => {
    try {
      const data = await listWorkerSessions();
      const all = data.workers || [];
      setWorkers(all.filter(w => w.status === 'online'));
    } catch (e) {
      console.error('Failed to load workers:', e);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    // Load DB stats for rebuild banner
    import('../../services/bridge').then(({ getDbStats }) => {
      getDbStats().then(d => setDbStats(d)).catch(() => {});
    });
    // Load auto_processing config once
    getAutoProcessing().then(data => {
      if (data.enabled != null) setAutoProcessing(data.enabled);
      if (data.rest_after_batch_s != null) setRestAfterBatch(data.rest_after_batch_s);
      if (data.batch_size != null) setBatchSize(data.batch_size);
      if (data.verbose_log != null) setVerboseLog(data.verbose_log);
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

      {/* Data rebuild banner (moved from SearchPanel) */}
      {dbStats?.build_status?.needs_rebuild && (
        <div className="mb-4 px-3 py-2.5 bg-amber-900/25 border border-amber-700 rounded-lg">
          <div className="text-amber-300 text-sm font-bold">{t('status.rebuild_required_title')}</div>
          <div className="text-amber-200/90 text-xs mt-1">
            {t('status.rebuild_required_desc', {
              db: dbStats.build_status.db_data_build_level ?? 0,
              current: dbStats.build_status.current_data_build_level ?? 0
            })}
          </div>
          {Array.isArray(dbStats.build_status.reasons) && dbStats.build_status.reasons.length > 0 && (
            <div className="mt-2 space-y-1">
              {dbStats.build_status.reasons.slice(0, 3).map((r, i) => (
                <div key={i} className="text-amber-100 text-[11px]">- {r}</div>
              ))}
            </div>
          )}
          <div className="text-amber-300 text-[11px] mt-2">{t('status.rebuild_required_action')}</div>
        </div>
      )}

      {/* Server Auto-Processing (unified: auto_processing + embedded worker) */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-gray-300">{t('worker.auto_title')}</div>
            <div className="text-xs text-gray-500 mt-0.5">{t('worker.auto_desc')}</div>
          </div>
          <div className="flex items-center gap-3">
            {autoProcessing && (
              <span className={`text-xs px-2 py-0.5 rounded ${
                embeddedStatus.running
                  ? 'bg-green-900/50 text-green-400'
                  : 'bg-yellow-900/50 text-yellow-400'
              }`}>
                {embeddedStatus.running ? t('worker.status_running') : embeddedStatus.status}
                {embeddedStatus.current_phase && ` · ${embeddedStatus.current_phase.toUpperCase()}`}
                {embeddedStatus.current_file && ` · ${embeddedStatus.current_file}`}
                {embeddedStatus.batch_throughput > 0 && ` · ${embeddedStatus.batch_throughput} f/m`}
                {embeddedStatus.jobs_completed > 0 && ` · ${embeddedStatus.jobs_completed} ${t('worker.jobs_completed')}`}
              </span>
            )}
            <button
              onClick={async () => {
                const newVal = !autoProcessing;
                setAutoProcessing(newVal);
                setEmbeddedEnabled(newVal);
                try { await updateAutoProcessing({ enabled: newVal }); } catch (e) { console.error(e); setAutoProcessing(!newVal); setEmbeddedEnabled(!newVal); }
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
        </div>
        {autoProcessing && (
          <div className="flex items-center gap-6 mt-3 pt-3 border-t border-gray-700/50">
            <div className="flex items-center gap-2">
              <div className="text-xs text-gray-400">{t('label.chunk_size')}</div>
              <input
                type="number" min="1" max="20" value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 5)}
                className="w-16 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white"
              />
              <span className="text-xs text-gray-500">files</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="text-xs text-gray-400">{t('worker.rest_title')}</div>
              <input
                type="number" min="0" max="300" value={restAfterBatch}
                onChange={(e) => setRestAfterBatch(parseInt(e.target.value) || 0)}
                className="w-16 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white"
              />
              <span className="text-xs text-gray-500">{t('worker.rest_unit')}</span>
            </div>
            <button
              onClick={async () => {
                const bs = Math.max(1, Math.min(20, batchSize));
                const rest = Math.max(0, Math.min(300, restAfterBatch));
                setBatchSize(bs);
                setRestAfterBatch(rest);
                try {
                  await updateAutoProcessing({ batch_size: bs, rest_after_batch_s: rest });
                } catch (err) { console.error(err); }
              }}
              className="px-3 py-1 text-xs font-medium rounded bg-purple-600 text-white hover:bg-purple-500 transition-colors"
            >
              {t('action.apply')}
            </button>
            <label className="flex items-center gap-1.5 ml-auto cursor-pointer">
              <input
                type="checkbox" checked={verboseLog}
                onChange={async (e) => {
                  const v = e.target.checked;
                  setVerboseLog(v);
                  try { await updateAutoProcessing({ verbose_log: v }); } catch (err) { console.error(err); }
                }}
                className="w-3 h-3 rounded border-gray-600 bg-gray-700 text-purple-500 focus:ring-0"
              />
              <span className="text-xs text-gray-500">상세 로그</span>
            </label>
          </div>
        )}
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
                  {(() => {
                    const mode = w.assigned_mode || w.processing_mode_override || 'full';
                    const isOverride = !!w.processing_mode_override;
                    const badges = {
                      parse_thumb: { bg: 'bg-sky-900/50 text-sky-400', label: 'Parse' },
                      mc: { bg: 'bg-purple-900/50 text-purple-400', label: 'MC' },
                      mc_only: { bg: 'bg-purple-900/50 text-purple-400', label: 'MC' },
                      vv: { bg: 'bg-blue-900/50 text-blue-400', label: 'VV' },
                      mv: { bg: 'bg-green-900/50 text-green-400', label: 'MV' },
                      full: { bg: 'bg-gray-700/50 text-gray-300', label: 'Full' },
                      embed_only: { bg: 'bg-orange-900/50 text-orange-400', label: 'Embed' },
                    };
                    const b = badges[mode] || badges.full;
                    return (
                      <div className="flex items-center gap-1">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${b.bg}`}>
                          {b.label}
                        </span>
                        {w.phase_job_count > 0 && (
                          <span className="text-[10px] text-gray-500 font-mono">{w.phase_job_count}</span>
                        )}
                        {isOverride && (
                          <span className="text-[9px] text-yellow-600" title="Admin override">pin</span>
                        )}
                      </div>
                    );
                  })()}
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
                <td className="px-4 py-3 text-xs">
                  {w.current_phase ? (
                    <div className="flex items-center gap-1.5">
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                        w.current_phase === 'vision' || w.current_phase === 'mc' ? 'bg-purple-900/40 text-purple-300' :
                        w.current_phase === 'embed_vv' || w.current_phase === 'vv' ? 'bg-blue-900/40 text-blue-300' :
                        w.current_phase === 'embed_mv' || w.current_phase === 'mv' ? 'bg-green-900/40 text-green-300' :
                        w.current_phase === 'parse' ? 'bg-sky-900/40 text-sky-300' :
                        'bg-gray-700/40 text-gray-300'
                      }`}>
                        {w.current_phase === 'vision' ? 'MC' :
                         w.current_phase === 'embed_vv' ? 'VV' :
                         w.current_phase === 'embed_mv' ? 'MV' :
                         w.current_phase.toUpperCase()}
                      </span>
                      {w.current_file && (
                        <span className="text-gray-500 truncate max-w-[120px]" title={w.current_file}>
                          {w.current_file.split(/[/\\]/).pop()}
                        </span>
                      )}
                    </div>
                  ) : <span className="text-gray-600">-</span>}
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
