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
import { getJobStats } from '../../api/worker';
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
  const [queueStats, setQueueStats] = useState(null);

  const load = useCallback(async () => {
    try {
      const [workerData, statsData] = await Promise.all([
        listWorkerSessions(),
        getJobStats().catch(() => null),
      ]);
      const all = workerData.workers || [];
      setWorkers(all.filter(w => w.status === 'online'));
      if (statsData) setQueueStats(statsData);
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

      {/* Pipeline phase dashboard */}
      {onlineCount > 0 && (
        <div className="mb-4 space-y-3">
          {/* System performance dashboard */}
          {queueStats && (() => {
            const s = queueStats;
            const total = s.total || 0;
            const done = s.completed || 0;
            const failed = s.failed || 0;
            const inProgress = (s.assigned || 0) + (s.processing || 0);
            const waiting = (s.pending || 0);
            const pct = total > 0 ? ((done / total) * 100).toFixed(1) : 0;
            const speed = s.throughput || 0;
            const eta = s.eta_seconds;
            const etaStr = eta ? (eta >= 3600 ? `${Math.floor(eta/3600)}h ${Math.floor((eta%3600)/60)}m` : `${Math.floor(eta/60)}m`) : null;
            const secPerFile = speed > 0 ? Math.round(60 / speed) : null;

            // Pipeline position: each file counted in exactly ONE stage
            const phases = [
              { name: 'Download', pending: s.pipe_download ?? s.download_waiting ?? 0, color: 'yellow', speed: null },
              { name: 'Parse', pending: s.pipe_parse ?? s.parse_pending ?? 0, color: 'sky', speed: s.parse_throughput },
              { name: 'MC', pending: s.pipe_mc ?? s.mc_pending ?? 0, color: 'purple', speed: s.mc_throughput },
              { name: 'VV', pending: s.pipe_vv ?? 0, color: 'blue', speed: s.vv_throughput },
              { name: 'MV', pending: s.pipe_mv ?? 0, color: 'green', speed: s.mv_throughput },
            ];
            const workerPhases = phases.filter(p => p.name !== 'Download');
            const bottleneck = workerPhases.filter(p => p.pending > 0).reduce((a, b) => a.pending > b.pending ? a : b, workerPhases[0]);

            return (
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-3">
                {/* Row 1: Queue summary — total / in-progress / done */}
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-6 flex-1">
                    <div className="text-center">
                      <div className="text-2xl font-bold font-mono text-white">{total}</div>
                      <div className="text-[10px] text-gray-500">전체 큐</div>
                    </div>
                    <div className="text-gray-600">→</div>
                    <div className="text-center">
                      <div className={`text-2xl font-bold font-mono ${inProgress > 0 ? 'text-cyan-400' : 'text-gray-600'}`}>{inProgress}</div>
                      <div className="text-[10px] text-gray-500">진행 중</div>
                    </div>
                    <div className="text-gray-600">→</div>
                    <div className="text-center">
                      <div className={`text-2xl font-bold font-mono ${done > 0 ? 'text-emerald-400' : 'text-gray-600'}`}>{done}</div>
                      <div className="text-[10px] text-gray-500">완료</div>
                    </div>
                    {failed > 0 && <>
                      <div className="text-gray-600">|</div>
                      <div className="text-center">
                        <div className="text-2xl font-bold font-mono text-red-400">{failed}</div>
                        <div className="text-[10px] text-red-400/60">실패</div>
                      </div>
                    </>}
                  </div>

                  {/* Progress bar */}
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-400">{pct}%</span>
                    </div>
                    <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500 rounded-full transition-all duration-500"
                        style={{ width: `${pct}%` }} />
                    </div>
                  </div>

                  {/* Speed / ETA */}
                  <div className="text-center shrink-0">
                    <div className={`text-2xl font-bold font-mono ${speed > 0 ? 'text-emerald-400' : 'text-gray-600'}`}>
                      {speed > 0 ? speed.toFixed(1) : '-'}
                    </div>
                    <div className="text-[10px] text-gray-500">files/min</div>
                  </div>

                  {/* Per-file time */}
                  <div className="text-center shrink-0">
                    <div className={`text-2xl font-bold font-mono ${secPerFile ? 'text-amber-400' : 'text-gray-600'}`}>
                      {secPerFile ? (secPerFile >= 60 ? `${Math.floor(secPerFile/60)}m${secPerFile%60}s` : `${secPerFile}s`) : '-'}
                    </div>
                    <div className="text-[10px] text-gray-500">/file</div>
                  </div>

                  {/* ETA */}
                  <div className="text-center shrink-0">
                    <div className={`text-2xl font-bold font-mono ${etaStr ? 'text-white' : 'text-gray-600'}`}>
                      {etaStr || '-'}
                    </div>
                    <div className="text-[10px] text-gray-500">남은시간</div>
                  </div>

                  {/* Workers */}
                  <div className="text-center shrink-0">
                    <div className="text-2xl font-bold font-mono text-green-400">{onlineCount}</div>
                    <div className="text-[10px] text-gray-500">workers</div>
                  </div>
                </div>

                {/* Row 2: Phase pipeline — pending + speed + bottleneck */}
                <div className="flex items-stretch gap-0.5">
                  {phases.map((p, i) => {
                    const isBn = p.name !== 'Download' && p === bottleneck && p.pending > 0;
                    const phaseSpeed = p.speed;
                    const phaseSecPer = phaseSpeed > 0 ? Math.round(60 / phaseSpeed) : null;
                    return (
                      <div key={p.name} className="flex items-center flex-1">
                        <div className={`flex-1 text-center py-2 rounded ${
                          isBn ? 'bg-red-900/20 border border-red-700/50' : 'bg-gray-750'
                        }`}>
                          <div className="flex items-center justify-center gap-1 mb-0.5">
                            <span className={`text-[10px] font-medium ${p.pending > 0 ? `text-${p.color}-400` : 'text-gray-600'}`}>{p.name}</span>
                            {isBn && <span className="text-[8px] text-red-400">bottleneck</span>}
                          </div>
                          <div className={`text-lg font-bold font-mono ${p.pending > 0 ? `text-${p.color}-400` : 'text-gray-700'}`}>
                            {p.pending}
                          </div>
                          {p.pending > 0 && <div className="text-[8px] text-gray-500">대기</div>}
                          {phaseSpeed > 0 && (
                            <div className="text-[9px] text-gray-400 font-mono">
                              {phaseSpeed}/m {phaseSecPer && <span className="text-gray-500">· {phaseSecPer}s</span>}
                            </div>
                          )}
                        </div>
                        {i < phases.length - 1 && <span className="text-gray-700 mx-0.5 text-xs">→</span>}
                      </div>
                    );
                  })}
                  <span className="text-gray-700 mx-0.5 text-xs self-center">→</span>
                  <div className="text-center py-2 flex-1">
                    <div className="text-[10px] text-emerald-400/60 mb-0.5">Done</div>
                    <div className="text-lg font-bold font-mono text-emerald-400">{done}</div>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* Worker table — phase columns with count + speed */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400 text-xs">
              <th className="text-left px-3 py-2">{t('admin.worker_name')}</th>
              <th className="text-center px-2 py-2">
                <div className="text-sky-400">Parse</div>
                {queueStats?.parse_throughput > 0 && <div className="text-[9px] text-sky-400/60 font-mono">{queueStats.parse_throughput}/m</div>}
              </th>
              <th className="text-center px-2 py-2">
                <div className="text-purple-400">MC</div>
                {queueStats?.mc_throughput > 0 && <div className="text-[9px] text-purple-400/60 font-mono">{queueStats.mc_throughput}/m</div>}
              </th>
              <th className="text-center px-2 py-2">
                <div className="text-blue-400">VV</div>
                {queueStats?.vv_throughput > 0 && <div className="text-[9px] text-blue-400/60 font-mono">{queueStats.vv_throughput}/m</div>}
              </th>
              <th className="text-center px-2 py-2">
                <div className="text-green-400">MV</div>
                {queueStats?.mv_throughput > 0 && <div className="text-[9px] text-green-400/60 font-mono">{queueStats.mv_throughput}/m</div>}
              </th>
              <th className="text-center px-2 py-2">Batch</th>
              <th className="text-right px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {workers.map((w) => {
              const pc = w.resources?.phase_counts || {};
              const cur = w.current_phase === 'vision' ? 'mc' :
                w.current_phase === 'embed_vv' ? 'vv' :
                w.current_phase === 'embed_mv' ? 'mv' :
                w.current_phase === 'parse' ? 'parse' : null;
              const batchSize = w.batch_capacity_override || w.batch_capacity;
              const fileName = w.current_file ? w.current_file.split(/[/\\]/).pop() : null;

              const phaseLabel = cur === 'mc' ? 'MC' : cur === 'vv' ? 'VV' : cur === 'mv' ? 'MV' : cur === 'parse' ? 'Parse' : null;

              const phaseCell = (key, color) => {
                const count = pc[key] || 0;
                const isCurrent = cur === key;
                return (
                  <td className={`text-center px-2 py-2 ${isCurrent ? `bg-${color}-900/30` : ''}`}>
                    <div className={`text-sm font-bold font-mono ${count > 0 ? `text-${color}-400` : 'text-gray-700'}`}>
                      {count || '-'}
                    </div>
                    {isCurrent && w.throughput > 0 && (
                      <div className="text-[9px] text-emerald-400 font-mono">{w.throughput.toFixed(1)}/m</div>
                    )}
                    {isCurrent && fileName && (
                      <div className={`text-[9px] text-${color}-300 truncate max-w-[120px] mx-auto`} title={w.current_file}>
                        {fileName}
                      </div>
                    )}
                  </td>
                );
              };

              return (
                <tr key={w.id} className="border-b border-gray-700/50 hover:bg-gray-700/20">
                  <td className="px-3 py-2">
                    <div className="text-xs font-medium text-white">
                      {w.worker_name === '__builtin__' ? t('admin.worker_builtin_label') : w.worker_name}
                    </div>
                    <div className="text-[10px] text-gray-500">{w.hostname}</div>
                  </td>
                  {w.worker_name === '__builtin__' ? phaseCell('parse', 'sky') : <td className="text-center px-2 py-2 text-gray-800">-</td>}
                  {phaseCell('mc', 'purple')}
                  {phaseCell('vv', 'blue')}
                  {phaseCell('mv', 'green')}
                  <td className="text-center px-2 py-2">
                    {editingCapacity?.id === w.id ? (
                      <input type="number" min={1} max={32} value={editingCapacity.value}
                        onChange={(e) => setEditingCapacity({ id: w.id, value: Math.max(1, Math.min(32, Number(e.target.value))) })}
                        onKeyDown={(e) => { if (e.key === 'Enter') handleCapacitySave(w.id); if (e.key === 'Escape') setEditingCapacity(null); }}
                        onBlur={() => handleCapacitySave(w.id)} autoFocus
                        className="w-10 bg-gray-900 border border-blue-500 rounded px-1 py-0 text-xs font-mono text-yellow-300 focus:outline-none text-center" />
                    ) : (
                      <span className="text-xs font-mono text-yellow-300 cursor-pointer hover:text-yellow-200"
                        onClick={() => setEditingCapacity({ id: w.id, value: batchSize })}>
                        {batchSize}
                      </span>
                    )}
                  </td>
                  <td className="text-right px-3 py-2">
                    {w.status === 'online' && (
                      <div className="flex gap-0.5 justify-end">
                        <button onClick={() => handleStop(w.id)}
                          className="p-1 rounded hover:bg-yellow-900/50 text-gray-500 hover:text-yellow-300">
                          <Square size={10} />
                        </button>
                        <button onClick={() => handleBlock(w.id)}
                          className="p-1 rounded hover:bg-red-900/50 text-gray-500 hover:text-red-300">
                          <Ban size={10} />
                        </button>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {workers.length === 0 && (
          <div className="text-center text-gray-500 py-8 text-sm">{t('admin.workers_empty')}</div>
        )}
      </div>
    </div>
  );
}
