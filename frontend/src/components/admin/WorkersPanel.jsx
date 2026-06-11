/**
 * WorkersPanel — worker session management + auto-processing config.
 * Extracted from AdminPage.jsx.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import {
  listWorkerSessions, stopWorkerSession, blockWorkerSession, unblockWorkerSession,
  getAutoProcessing, updateAutoProcessing,
  getEmbeddedWorker,
  getPausedPhases, setPausedPhases,
} from '../../api/admin';
import {
  listAnalysisJobs, pauseAnalysisJob, resumeAnalysisJob,
  cancelAnalysisJob, retryFailedTasks, getAnalysisMetrics, archiveAnalysisJob,
  getJobErrors,
} from '../../api/analysis';
import AnalysisJobCard from '../AnalysisJobCard';
// Legacy getJobStats removed — using analysis metrics API
import {
  RefreshCw, Square, Ban, X,
} from 'lucide-react';

function sumFailedPhases(failed) {
  if (!failed || typeof failed !== 'object') return 0;
  return Object.values(failed).reduce((sum, value) => sum + (value || 0), 0);
}

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
  const [blockedWorkers, setBlockedWorkers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [autoProcessing, setAutoProcessing] = useState(true);
  const [verboseLog, setVerboseLog] = useState(false);
  const [embeddedStatus, setEmbeddedStatus] = useState({ running: false, status: 'idle', jobs_completed: 0 });
  const [dbStats, setDbStats] = useState(null);
  const [queueStats, setQueueStats] = useState(null);
  const [analysisJobs, setAnalysisJobs] = useState([]);
  const [pausedPhases, setPausedPhasesState] = useState({ dl: false, parse: false, mc: false, vv: false, mv: false });

  const load = useCallback(async () => {
    try {
      const [workerData, jobsData, pausedData] = await Promise.all([
        listWorkerSessions(),
        listAnalysisJobs(true).catch(() => ({ jobs: [] })),
        getPausedPhases().catch(() => null),
      ]);
      if (pausedData) setPausedPhasesState(pausedData);
      let statsData = {};
      const all = workerData.workers || [];
      setWorkers(all.filter(w => w.status === 'online'));
      setBlockedWorkers(all.filter(w => w.status === 'blocked'));
      const jobs = (jobsData.jobs || []).filter(j => j.status !== 'cancelled');
      setAnalysisJobs(jobs);

      // Fetch metrics for active jobs (phase-level throughput)
      const activeJobs = jobs.filter(j => j.status === 'active' || j.status === 'paused');
      if (activeJobs.length > 0) {
        try {
          const metricsResults = await Promise.all(
            activeJobs.map(j => getAnalysisMetrics(j.id).catch(() => null))
          );
          // Merge phase_metrics into statsData for display
          if (statsData) {
            for (const m of metricsResults) {
              if (!m) continue;
              const t = m.throughput || {};
              if (t.files_per_min > 0) statsData.throughput = t.files_per_min;
              if (t.eta_seconds) statsData.eta_seconds = t.eta_seconds;
              const pm = m.phase_metrics || {};
              // Use fpm (direct) or convert avg_s to files/min
              statsData.mc_throughput = pm.mc?.fpm || (pm.mc?.avg_s > 0 ? Math.round(60 / pm.mc.avg_s * 10) / 10 : 0);
              statsData.vv_throughput = pm.vv?.fpm || (pm.vv?.avg_s > 0 ? Math.round(60 / pm.vv.avg_s * 10) / 10 : 0);
              statsData.mv_throughput = pm.mv?.fpm || (pm.mv?.avg_s > 0 ? Math.round(60 / pm.mv.avg_s * 10) / 10 : 0);
            }
          }
        } catch {
          // Metrics are advisory; the worker list still renders without them.
        }
      }

      if (statsData && activeJobs.length > 0) {
        const totals = activeJobs.reduce((acc, j) => {
          const p = j.progress || {};
          acc.total += p.total || 0;
          acc.complete += p.complete || 0;
          acc.failed += sumFailedPhases(p.failed);
          return acc;
        }, { total: 0, complete: 0, failed: 0 });
        statsData.total = totals.total;
        statsData.completed = totals.complete;
        statsData.failed = totals.failed;
        statsData.pending = Math.max(totals.total - totals.complete - totals.failed, 0);
      }
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
      if (data.verbose_log != null) setVerboseLog(data.verbose_log);
    }).catch(() => {});
    // Load embedded worker status
    const loadEmbedded = () => {
      getEmbeddedWorker().then(data => {
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

  const onlineCount = workers.filter(w => w.status === 'online').length;

  if (loading) return <div className="text-gray-400 text-sm">{t('status.loading')}</div>;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex items-center justify-between mb-4 shrink-0">
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

      {/* Scrollable content area */}
      <div className="flex-1 overflow-y-auto min-h-0 space-y-4">

      {/* Server Auto-Processing (unified: auto_processing + embedded worker) */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
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
        </div>
        {autoProcessing && (
          <div className="flex items-center gap-4 mt-3 pt-3 border-t border-gray-700/50">
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

      {/* Unified dashboard: summary + queue list */}
      {analysisJobs.length > 0 && (() => {
        const allJobs = analysisJobs;
        // Totals across all jobs
        const totals = allJobs.reduce((acc, j) => {
          const p = j.progress || {};
          acc.total += p.total || 0;
          acc.complete += p.complete || 0;
          acc.dl += p.downloaded || 0;
          acc.parse += p.parsed || 0;
          acc.mc += p.mc_done || 0;
          acc.vv += p.vv_done || 0;
          acc.mv += p.mv_done || 0;
          acc.failed += sumFailedPhases(p.failed);
          return acc;
        }, { total: 0, complete: 0, dl: 0, parse: 0, mc: 0, vv: 0, mv: 0, failed: 0 });
        const pct = totals.total > 0 ? (totals.complete / totals.total * 100).toFixed(1) : 0;
        const s = queueStats || {};
        const speed = s.throughput || 0;
        const eta = s.eta_seconds;
        const etaStr = eta ? (eta >= 3600 ? `${Math.floor(eta/3600)}h${Math.floor((eta%3600)/60)}m` : `${Math.floor(eta/60)}m`) : null;
        const secPer = speed > 0 ? Math.round(60 / speed) : null;

        return (
          <div className="mb-4 rounded-xl bg-gray-800/60 border border-gray-700/40 overflow-hidden">
            {/* Summary header */}
            <div className="px-4 py-3 space-y-2">
              {/* Row 1: counts + progress + speed */}
              <div className="flex items-center gap-3">
                <span className="text-2xl font-bold font-mono text-emerald-400">{totals.complete}</span>
                <span className="text-gray-600 text-lg">/</span>
                <span className="text-2xl font-mono text-gray-300">{totals.total}</span>
                <div className="flex-1 mx-2 h-2.5 bg-gray-700 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 rounded-full transition-all" style={{ width: `${pct}%` }} />
                </div>
                <span className="text-lg font-bold font-mono text-blue-400">{pct}%</span>
              </div>
              {/* Row 2: Phase table — 항목 / 처리량 / 속도 (세로 3단) */}
              <div className="grid grid-cols-5 gap-1 text-[10px] font-mono text-center">
                {/* Header */}
                {['DL', 'Parse', 'MC', 'MV', 'VV'].map(h => (
                  <span key={h} className="text-gray-500 font-semibold">{h}</span>
                ))}
                {/* 처리량 (done/total) */}
                {[
                  [totals.dl, 'text-yellow-400'],
                  [totals.parse, 'text-sky-400'],
                  [totals.mc, 'text-purple-400'],
                  [totals.mv, 'text-green-400'],
                  [totals.vv, 'text-cyan-400'],
                ].map(([done, color], i) => (
                  <span key={i} className={color}>
                    {done}<span className="text-gray-600">/{totals.total}</span>
                  </span>
                ))}
                {/* 속도 (files/min) */}
                {[
                  s.download_throughput,
                  s.parse_throughput,
                  s.mc_throughput,
                  s.mv_throughput,
                  s.vv_throughput,
                ].map((spd, i) => (
                  <span key={i} className={spd > 0 ? 'text-emerald-400' : 'text-gray-700'}>
                    {spd > 0 ? `${spd}/m` : '-'}
                  </span>
                ))}
                {/* ON/OFF 토글 */}
                {['dl', 'parse', 'mc', 'mv', 'vv'].map(phase => {
                  const paused = pausedPhases[phase];
                  return (
                    <button
                      key={phase}
                      onClick={async () => {
                        const next = !paused;
                        setPausedPhasesState(prev => ({ ...prev, [phase]: next }));
                        try { await setPausedPhases({ [phase]: next }); } catch { setPausedPhasesState(prev => ({ ...prev, [phase]: !next })); }
                      }}
                      className={`px-1.5 py-0.5 rounded text-[9px] font-bold transition-colors ${
                        paused
                          ? 'bg-red-900/40 text-red-400 hover:bg-red-800/50'
                          : 'bg-emerald-900/30 text-emerald-400 hover:bg-emerald-800/40'
                      }`}
                    >
                      {paused ? 'OFF' : 'ON'}
                    </button>
                  );
                })}
              </div>
              {/* Row 3: speed detail + action buttons */}
              <div className="flex items-center gap-3 pt-1 border-t border-gray-700/20">
                {/* Speed metrics */}
                <div className="flex items-center gap-4 flex-1 text-xs font-mono">
                  {speed > 0 ? (
                    <>
                      <span className="text-emerald-400 font-bold">{speed.toFixed(1)} files/min</span>
                      {secPer && <span className="text-amber-400">{secPer}s/file</span>}
                      {etaStr && <span className="text-white">ETA ~{etaStr}</span>}
                    </>
                  ) : (
                    <span className="text-gray-600">대기 중...</span>
                  )}
                  <span className="text-green-400">{onlineCount} workers</span>
                </div>
                {/* Action buttons */}
                <div className="flex items-center gap-2">
                  {totals.failed > 0 && (
                    <button onClick={async () => {
                      try {
                        for (const j of allJobs.filter(j => j.status === 'active')) {
                          await retryFailedTasks(j.id);
                        }
                        load();
                        alert('실패 파일 재시도 요청 완료');
                      } catch { alert('재시도 실패'); }
                    }}
                      className="px-3 py-1 rounded bg-blue-900/30 text-blue-400 hover:bg-blue-900/50 text-[10px] font-medium">
                      실패 재시도
                    </button>
                  )}
                  <button onClick={async () => {
                    if (!confirm('모든 활성 분석작업을 일시정지하시겠습니까?')) return;
                    try {
                      for (const j of allJobs.filter(j => j.status === 'active')) {
                        await pauseAnalysisJob(j.id);
                      }
                      load();
                    } catch { alert('중지 실패'); }
                  }}
                    className="px-3 py-1 rounded bg-red-900/30 text-red-400 hover:bg-red-900/50 text-[10px] font-medium">
                    전체 중지
                  </button>
                </div>
              </div>
            </div>
            {/* Queue list — click to expand details */}
            <div className="border-t border-gray-700/30 px-4 py-2 space-y-0.5">
              {allJobs.map(job => <QueueRow key={job.id} job={job} onAction={async (id, action) => {
                try {
                  if (action === 'pause') await pauseAnalysisJob(id);
                  else if (action === 'resume') await resumeAnalysisJob(id);
                  else if (action === 'cancel') await cancelAnalysisJob(id);
                  else if (action === 'retry') await retryFailedTasks(id);
                  else if (action === 'archive') await archiveAnalysisJob(id);
                  load();
                } catch (e) { console.error(`Job ${action} failed:`, e); }
              }} />)}
            </div>
          </div>
        );
      })()}

      {/* Worker table — phase columns with count + speed */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700 text-gray-400 text-xs">
              <th className="text-left px-3 py-2">{t('admin.worker_name')}</th>
              <th className="text-center px-2 py-2">
                <div className="text-purple-400">MC</div>
              </th>
              <th className="text-center px-2 py-2">
                <div className="text-green-400">MV</div>
              </th>
              <th className="text-center px-2 py-2">
                <div className="text-blue-400">VV</div>
              </th>
              <th className="text-center px-2 py-2">배치</th>
              <th className="text-right px-3 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {workers.map((w) => {
              const pc = w.phase_counts || w.resources?.phase_counts || {};
              const cur = (w.current_phase === 'vision' || w.current_phase === 'mc') ? 'mc' :
                (w.current_phase === 'embed_vv' || w.current_phase === 'vv') ? 'vv' :
                (w.current_phase === 'embed_mv' || w.current_phase === 'mv') ? 'mv' :
                w.current_phase === 'parse' ? 'parse' : null;
              const fileName = w.current_file ? w.current_file.split(/[/\\]/).pop() : null;

              const phaseLabel = cur === 'mc' ? 'MC' : cur === 'vv' ? 'VV' : cur === 'mv' ? 'MV' : cur === 'parse' ? 'Parse' : null;
              const batchCount = w.batch_capacity || 0;

              const pt = w.resources?.phase_throughput || {};
              // mc_speed=0 in scores means incapable
              const scores = w.resources?.scores?.phases || {};
              const isIncapable = (key) => {
                const s = scores[key];
                return s && s.status === 'incapable';
              };

              const phaseCell = (key, color) => {
                const isCurrent = cur === key;
                const speed = pt[key] || 0;
                const count = pc[key] || 0;
                const disabled = isIncapable(key);
                return (
                  <td className={`text-center px-2 py-1 ${isCurrent ? `bg-${color}-900/20` : ''} ${disabled ? 'opacity-30' : ''}`}>
                    {/* Row 1: 처리량 or 처리중 표시 */}
                    <div className={`text-xs font-mono ${disabled ? 'text-red-400' : isCurrent ? `text-${color}-400 font-bold` : 'text-gray-600'}`}>
                      {disabled ? '✕' : count > 0 ? count : isCurrent ? '...' : '-'}
                    </div>
                    {/* Row 2: 속도 */}
                    <div className={`text-[9px] font-mono ${isCurrent ? 'text-emerald-400' : 'text-gray-600'}`}>
                      {disabled ? '' : speed > 0 ? `${speed}/m` : isCurrent ? '' : '-'}
                    </div>
                    {/* Row 3: 파일명 (MC만) */}
                    {isCurrent && key === 'mc' && fileName && (
                      <div className={`text-[8px] text-${color}-300/70 truncate max-w-[120px] mx-auto`} title={w.current_file}>
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
                      {w.origin === 'server-local'
                        ? t('admin.worker_builtin_label') : w.worker_name}
                    </div>
                    <div className="text-[10px] text-gray-500">{w.hostname}</div>
                    {(w.origin || w.launcher) && (
                      <div className="text-[9px] text-gray-500 mt-0.5 font-mono">
                        {w.origin}{w.origin && w.launcher ? ' · ' : ''}{w.launcher}
                      </div>
                    )}
                  </td>
                  {phaseCell('mc', 'purple')}
                  {phaseCell('mv', 'green')}
                  {phaseCell('vv', 'blue')}
                  <td className="text-center px-2 py-1">
                    {cur && batchCount > 0 ? (
                      <div className="text-xs font-bold font-mono text-yellow-300">{batchCount}</div>
                    ) : w.current_phase === 'resting' ? (
                      <div className="text-[9px] text-orange-400 animate-pulse">휴식</div>
                    ) : (
                      <span className="text-gray-600 text-[9px]">-</span>
                    )}
                    {cur && (
                      <div className="text-[9px] text-gray-500">{phaseLabel}</div>
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

      {/* Blocked workers — durable until admin unblocks */}
      {blockedWorkers.length > 0 && (
        <div className="bg-red-950/20 rounded-lg border border-red-900/40 p-3">
          <div className="text-xs font-medium text-red-300 mb-2">{t('admin.worker_blocked_title')}</div>
          <div className="space-y-1">
            {blockedWorkers.map((w) => (
              <div key={w.id} className="flex items-center justify-between text-xs">
                <span className="text-gray-300">
                  {w.worker_name}
                  <span className="text-gray-500 ml-2">{w.hostname}</span>
                </span>
                <button
                  onClick={async () => {
                    try { await unblockWorkerSession(w.id); load(); }
                    catch (e) { console.error('Unblock failed:', e); }
                  }}
                  className="px-2 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 text-[10px]"
                >
                  {t('admin.worker_unblock')}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Bottom spacer — prevent StatusBar from hiding last row */}
      <div className="h-16" />
      </div>{/* end scrollable content */}
    </div>
  );
}


// ── Queue Row (expandable) ─────────────────────────────────

/** Inline error detail — shows failed phase summary + expandable file list */
function ErrorDetail({ jobId, failed }) {
  const [errorList, setErrorList] = useState(null);
  const [open, setOpen] = useState(false);
  const [dismissing, setDismissing] = useState(false);

  const loadErrors = async () => {
    try {
      const data = await getJobErrors(jobId);
      setErrorList(data.errors || []);
    } catch {
      setErrorList([]);
    }
  };

  const toggle = async (e) => {
    e.stopPropagation();
    if (open) { setOpen(false); return; }
    if (!errorList) await loadErrors();
    setOpen(true);
  };

  const handleDismiss = async (e) => {
    e.stopPropagation();
    if (!confirm('영구 실패 파일을 확인 처리하시겠습니까?\n작업 목록에서 제거됩니다.')) return;
    setDismissing(true);
    try {
      const { dismissFailedTasks } = await import('../../api/analysis');
      await dismissFailedTasks(jobId);
      await loadErrors();
    } catch { /* ignore */ }
    setDismissing(false);
  };

  const hasPermanent = errorList?.some(e => e.permanent);

  return (
    <div>
      <div className="flex items-center gap-2">
        <span className="text-red-400">
          실패: {Object.entries(failed).filter(([, v]) => v > 0).map(([k, v]) => `${k}:${v}`).join(' · ')}
        </span>
        <button onClick={toggle} className="text-red-400/60 hover:text-red-300 underline text-[9px]">
          {open ? '닫기' : '상세'}
        </button>
      </div>
      {open && errorList && (
        <div className="mt-1 space-y-1">
          <div className="max-h-32 overflow-y-auto space-y-0.5">
            {errorList.length === 0 ? (
              <div className="text-gray-600">에러 없음 (모두 확인 처리됨)</div>
            ) : errorList.map((err, i) => (
              <div key={i} className="flex items-start gap-2 text-[9px]">
                <span className="text-red-400 shrink-0">[{err.failed_phases.join(',')}]</span>
                <span className="text-gray-400" title={err.file_name}>{err.file_name}</span>
                {err.permanent && <span className="text-red-600 shrink-0 font-bold">영구</span>}
                {err.error && <span className="text-gray-600 truncate" title={err.error}>— {err.error}</span>}
              </div>
            ))}
          </div>
          {hasPermanent && (
            <button
              onClick={handleDismiss}
              disabled={dismissing}
              className="px-2 py-0.5 rounded bg-gray-700 text-gray-300 hover:bg-gray-600 text-[9px] disabled:opacity-50"
            >
              {dismissing ? '처리 중...' : '영구 실패 확인 (목록에서 제거)'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function QueueRow({ job, onAction }) {
  const [expanded, setExpanded] = useState(false);
  const p = job.progress || {};
  const total = p.total || 0;
  const done = p.complete || 0;
  const pct = total > 0 ? (done / total * 100).toFixed(0) : 0;
  const isActive = job.status === 'active' || job.status === 'paused';
  const isPaused = job.status === 'paused';
  const failed = p.failed || {};
  const totalFailed = sumFailedPhases(failed);

  return (
    <div>
      {/* Summary row — click to expand */}
      <div className="flex items-center gap-2 py-1 text-xs font-mono cursor-pointer hover:bg-gray-700/20 rounded px-1 -mx-1 group"
           onClick={() => setExpanded(!expanded)}>
        <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${isActive ? 'bg-blue-500 animate-pulse' : 'bg-emerald-500'}`} />
        <span className="text-gray-300 truncate flex-1">{job.name}</span>
        <span className="text-gray-500 tabular-nums">{done}/{total}</span>
        <div className="w-16 h-1 bg-gray-700/50 rounded-full overflow-hidden flex-shrink-0">
          <div className={`h-full rounded-full ${isActive ? 'bg-blue-500' : 'bg-emerald-500'}`} style={{ width: `${pct}%` }} />
        </div>
        <span className="text-gray-500 w-8 text-right">{pct}%</span>
        {!isActive && onAction && (
          <button onClick={(e) => { e.stopPropagation(); onAction(job.id, 'archive'); }}
            className="p-0.5 rounded hover:bg-red-900/30 text-gray-600 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
            title="삭제">
            <X size={10} />
          </button>
        )}
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="ml-4 pl-3 border-l border-gray-700/30 py-2 space-y-1.5 text-[10px] font-mono">
          <div className="text-gray-600 truncate" title={job.source_path}>{job.source_path}</div>
          {/* All phases */}
          <div className="flex items-center gap-4">
            <span className="text-yellow-400">DL: {p.downloaded || 0}/{total}</span>
            <span className="text-sky-400">Parse: {p.parsed || 0}/{total}</span>
            <span className="text-purple-400">MC: {p.mc_done || 0}/{total}</span>
            <span className="text-cyan-400">VV: {p.vv_done || 0}/{total}</span>
            <span className="text-green-400">MV: {p.mv_done || 0}/{total}</span>
          </div>
          {/* Failures with detail toggle */}
          {totalFailed > 0 && (
            <ErrorDetail jobId={job.id} failed={failed} totalFailed={totalFailed} />
          )}
          {/* Controls */}
          {onAction && (
            <div className="flex items-center gap-2 pt-1">
              {isActive && !isPaused && (
                <button onClick={(e) => { e.stopPropagation(); onAction(job.id, 'pause'); }}
                  className="px-2 py-0.5 rounded bg-yellow-900/30 text-yellow-400 hover:bg-yellow-900/50 text-[9px]">
                  일시정지
                </button>
              )}
              {isActive && isPaused && (
                <button onClick={(e) => { e.stopPropagation(); onAction(job.id, 'resume'); }}
                  className="px-2 py-0.5 rounded bg-green-900/30 text-green-400 hover:bg-green-900/50 text-[9px]">
                  재개
                </button>
              )}
              {isActive && totalFailed > 0 && (
                <button onClick={(e) => { e.stopPropagation(); onAction(job.id, 'retry'); }}
                  className="px-2 py-0.5 rounded bg-blue-900/30 text-blue-400 hover:bg-blue-900/50 text-[9px]">
                  재시도
                </button>
              )}
              {isActive && (
                <button onClick={(e) => { e.stopPropagation(); onAction(job.id, 'cancel'); }}
                  className="px-2 py-0.5 rounded bg-red-900/30 text-red-400 hover:bg-red-900/50 text-[9px]">
                  취소
                </button>
              )}
              {!isActive && (
                <button onClick={(e) => { e.stopPropagation(); onAction(job.id, 'archive'); }}
                  className="px-2 py-0.5 rounded bg-red-900/30 text-red-400 hover:bg-red-900/50 text-[9px]">
                  삭제
                </button>
              )}
            </div>
          )}
          {job.created_at && (
            <div className="text-gray-700">생성: {new Date(job.created_at).toLocaleString()}</div>
          )}
        </div>
      )}
    </div>
  );
}
