/**
 * QueuePanel — job queue monitoring + work request management.
 * Extracted from AdminPage.jsx.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import {
  cleanupStaleJobs, cleanupQueue,
  getWorkRequests, getWorkRequestDetail, pauseWorkRequest, resumeWorkRequest, cancelWorkRequest,
  runRecoveryScan,
} from '../../api/admin';
import { getJobStats } from '../../api/worker';
import { getThumbnailStats } from '../../api/admin';
import { isElectron } from '../../api/client';
import {
  RefreshCw, AlertTriangle, Trash2, Folder, ChevronRight, Play, Loader2,
  Pause, CircleX,
} from 'lucide-react';


export default function QueuePanel() {
  const { t } = useLocale();
  const [stats, setStats] = useState(null);
  const [thumbStats, setThumbStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [cleanupMsg, setCleanupMsg] = useState('');
  const [workRequests, setWorkRequests] = useState([]);
  const [showCompleted, setShowCompleted] = useState(false);
  const [expandedWR, setExpandedWR] = useState(new Set());
  const [wrDetails, setWrDetails] = useState({});

  // IPC/HTTP dual-mode helpers
  const useIPC = isElectron && window.electron?.queue;

  const fetchWR = useCallback(async (includeCompleted) => {
    if (useIPC) {
      const res = await window.electron.queue.listWorkRequests(includeCompleted);
      return res?.work_requests || [];
    }
    return getWorkRequests(includeCompleted);
  }, [useIPC]);

  const fetchWRDetail = useCallback(async (wrId) => {
    if (useIPC) {
      const res = await window.electron.queue.getWorkRequestDetail(wrId);
      return res?.success !== false ? res : null;
    }
    return getWorkRequestDetail(wrId);
  }, [useIPC]);

  const load = useCallback(async () => {
    try {
      const [jobData, thumbData, wrData] = await Promise.all([
        useIPC ? window.electron.queue.getStats() : getJobStats(),
        useIPC ? Promise.resolve(null) : getThumbnailStats().catch(() => null),
        fetchWR(showCompleted).catch(() => []),
      ]);
      if (jobData && jobData.success !== false) setStats(jobData);
      if (thumbData && thumbData.success !== false) {
        setThumbStats(thumbData);
      }
      setWorkRequests(Array.isArray(wrData) ? wrData : (wrData?.work_requests || []));
    } catch (e) {
      console.error('Failed to load queue stats:', e);
    }
    setLoading(false);
  }, [showCompleted, useIPC, fetchWR]);

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
        const detail = await fetchWRDetail(wrId);
        setWrDetails(prev => ({ ...prev, [wrId]: detail }));
      } catch (e) {
        console.error('Failed to load WR detail:', e);
      }
    }
  };

  const handlePauseResume = async (wr) => {
    try {
      if (wr.status === 'paused') {
        useIPC ? await window.electron.queue.resumeWorkRequest(wr.id) : await resumeWorkRequest(wr.id);
      } else {
        useIPC ? await window.electron.queue.pauseWorkRequest(wr.id) : await pauseWorkRequest(wr.id);
      }
      load();
    } catch (e) {
      console.error('Pause/resume failed:', e);
    }
  };

  const handleCancelWR = async (wrId) => {
    try {
      useIPC ? await window.electron.queue.cancelWorkRequest(wrId) : await cancelWorkRequest(wrId);
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
