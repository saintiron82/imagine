import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Square, RefreshCw, Activity, AlertCircle, Clock, Loader2, Zap } from 'lucide-react';
import { useLocale } from '../i18n';
import { apiClient, isElectron, getServerUrl, getAccessToken, getRefreshToken } from '../api/client';
import { getJobStats } from '../api/worker';
import { PerformanceLimits } from '../pages/WorkerPage';

function formatEta(ms) {
    const sec = Math.ceil(ms / 1000);
    if (sec < 60) return `${sec}s`;
    const min = Math.floor(sec / 60);
    const remSec = sec % 60;
    if (min < 60) return `${min}m ${remSec}s`;
    const hr = Math.floor(min / 60);
    const remMin = min % 60;
    return `${hr}h ${remMin}m`;
}

export default function ClientWorkerView({ appMode, isWorkerRunning = false, workerProgress = {}, onWorkerStart, onWorkerStop }) {
  const { t } = useLocale();
  const [stats, setStats] = useState(null);
  const [logs, setLogs] = useState([]);
  const logEndRef = useRef(null);
  const pollRef = useRef(null);

  // Derive workerStatus from isWorkerRunning prop (App.jsx is the single source of truth)
  const workerStatus = isWorkerRunning ? 'running' : 'idle';

  const addLog = useCallback((message, type = 'info') => {
    const entry = { message, type, timestamp: new Date().toLocaleTimeString() };
    setLogs(prev => {
      const next = [...prev, entry];
      return next.length > 100 ? next.slice(-100) : next;
    });
  }, []);

  // Poll stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getJobStats();
        if (data.success !== false) setStats(data);
      } catch { /* ignore */ }
    };

    fetchStats();
    pollRef.current = setInterval(fetchStats, 5000);
    return () => clearInterval(pollRef.current);
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Worker log IPC listener only (status/jobDone handled by App.jsx)
  useEffect(() => {
    if (!isElectron) return;
    const w = window.electron?.worker;
    if (!w) return;

    const onLog = (data) => addLog(data.message, data.type);
    w.onLog?.(onLog);

    return () => {
      w.offLog?.();
    };
  }, [addLog]);

  const handleStart = async () => {
    if (isElectron) {
      if (!getServerUrl() || !getAccessToken()) {
        addLog('Not logged in. Please login first.', 'error');
        return;
      }
      addLog(t('worker.connecting'), 'info');
      await onWorkerStart?.();
    } else {
      try {
        const result = await apiClient.post('/api/v1/admin/worker/start');
        if (result.success === false) {
          addLog(result.error || 'Failed to start worker', 'error');
          return;
        }
        addLog(t('worker.connecting'), 'info');
      } catch (e) {
        addLog(e.detail || e.message, 'error');
      }
    }
  };

  const handleStop = async () => {
    if (isElectron) {
      try {
        addLog(t('worker.stop'), 'info');
        await onWorkerStop?.(); // App.jsx handles stop + state reset
      } catch (e) {
        addLog(e.message, 'error');
      }
    } else {
      try {
        const result = await apiClient.post('/api/v1/admin/worker/stop');
        addLog(t('worker.stop'), 'info');
        if (result.jobs_completed) {
          addLog(`${t('worker.jobs_completed')}: ${result.jobs_completed}`, 'success');
        }
      } catch (e) {
        addLog(e.detail || e.message, 'error');
      }
    }
  };

  const statusColor = {
    idle: 'text-gray-400',
    running: 'text-green-400',
    stopping: 'text-yellow-400',
    error: 'text-red-400',
  };

  const statusIcon = {
    idle: <Clock size={16} />,
    running: <Activity size={16} className="animate-pulse" />,
    stopping: <Loader2 size={16} className="animate-spin" />,
    error: <AlertCircle size={16} />,
  };

  const statItems = [
    { key: 'pending', value: stats?.pending, color: 'text-yellow-400' },
    { key: 'assigned', value: stats?.assigned, color: 'text-blue-400' },
    { key: 'processing', value: stats?.processing, color: 'text-cyan-400' },
    { key: 'completed', value: stats?.completed, color: 'text-green-400' },
    { key: 'failed', value: stats?.failed, color: 'text-red-400' },
    { key: 'total', value: stats?.total, color: 'text-gray-300' },
  ];

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <div className="max-w-4xl mx-auto space-y-6">

        {/* Header */}
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Zap size={22} className="text-emerald-400" />
            {t('worker.client_worker')}
          </h2>
          {getServerUrl() && (
            <p className="text-xs text-gray-500 mt-1">{t('worker.connected_to', { url: getServerUrl() })}</p>
          )}
        </div>

        {/* Worker Control (Electron only) */}
        {isElectron && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className={`flex items-center gap-2 ${statusColor[workerStatus] || 'text-gray-400'}`}>
                  {statusIcon[workerStatus] || <Clock size={16} />}
                  {t(`worker.status_${workerStatus}`)}
                </span>
                {workerStatus !== 'running' && getServerUrl() && (
                  <span className="text-xs text-gray-500 truncate max-w-[200px]">
                    {getServerUrl()}
                  </span>
                )}
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleStart}
                  disabled={workerStatus === 'running' || workerStatus === 'stopping' || !getAccessToken()}
                  className="flex items-center gap-1.5 px-4 py-2 rounded text-sm font-medium bg-green-700 hover:bg-green-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Play size={14} fill="currentColor" />
                  {t('worker.start')}
                </button>
                <button
                  onClick={handleStop}
                  disabled={workerStatus !== 'running'}
                  className="flex items-center gap-1.5 px-4 py-2 rounded text-sm font-medium bg-red-700 hover:bg-red-600 text-white disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Square size={14} />
                  {t('worker.stop')}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Web mode notice */}
        {!isElectron && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 text-center">
            <p className="text-sm text-gray-400">{t('worker.web_mode_notice')}</p>
          </div>
        )}

        {/* Performance Limits (Electron only) */}
        {isElectron && <PerformanceLimits t={t} />}

        {/* Queue Stats */}
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <RefreshCw size={14} className={stats ? '' : 'animate-spin'} />
            {t('worker.queue_stats')}
          </h3>
          {stats ? (
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
              {statItems.map(({ key, value, color }) => (
                <div key={key} className="text-center">
                  <div className={`text-lg font-bold ${color}`}>{value ?? 0}</div>
                  <div className="text-xs text-gray-500">{t(`admin.queue_${key}`)}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-gray-500">{t('status.loading')}</div>
          )}
        </div>

        {/* Detailed Progress Card (when running) */}
        {isWorkerRunning && (() => {
          const wp = workerProgress;
          const completed = wp.completed || 0;
          const totalQ = wp.totalQueue || 0;
          const pending = wp.pending || 0;
          const progressPct = totalQ > 0 ? Math.min(100, (completed / totalQ) * 100) : 0;
          const perMin = wp.throughput || 0;

          const isMcOnly = wp.processingMode === 'mc_only';
          const isEmbedOnly = wp.processingMode === 'embed_only';
          const phaseOrder = ['parse', 'vision', 'embed_vv', 'embed_mv'];
          const phaseConfig = {
            parse:    { label: t('status.phase.parse'), color: 'bg-cyan-500',   textColor: 'text-cyan-400' },
            vision:   { label: 'MC',                    color: 'bg-blue-500',   textColor: 'text-blue-400' },
            embed_vv: { label: 'VV',                    color: 'bg-purple-500', textColor: 'text-purple-400' },
            embed_mv: { label: 'MV',                    color: 'bg-green-500',  textColor: 'text-green-400' },
          };
          // Server-handled phases per mode — shown as dimmed "SVR" pills
          const serverPhases = isMcOnly
            ? new Set(['parse', 'embed_vv', 'embed_mv'])   // mc_only: worker does MC only
            : isEmbedOnly
            ? new Set(['parse', 'vision'])                  // embed_only: worker does VV+MV only
            : new Set();
          const currentIdx = phaseOrder.indexOf(wp.currentPhase);

          return (
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
                <Activity size={14} className="text-emerald-400 animate-pulse" />
                {t('worker.progress_title')}
              </h3>

              {/* Overall progress bar */}
              <div className="w-full h-2 bg-gray-700 rounded-full mb-4">
                <div className="h-full rounded-full bg-emerald-500 transition-all duration-500"
                  style={{ width: `${progressPct}%` }}
                />
              </div>

              {/* 4-Phase Independent Progress Bars with per-phase speed */}
              <div className="grid grid-cols-2 gap-3 mb-4">
                {phaseOrder.map((phase, idx) => {
                  const cfg = phaseConfig[phase];
                  const isServer = serverPhases.has(phase);
                  const isActive = !isServer && wp.currentPhase === phase;
                  const isDone = !isServer && currentIdx > idx;
                  const pct = isActive && wp.phaseCount > 0
                    ? Math.round((wp.phaseIndex / wp.phaseCount) * 100)
                    : isDone ? 100 : 0;
                  const fpm = wp.phaseFpm?.[phase] || 0;
                  const elapsed = wp.phaseElapsed?.[phase] || 0;

                  return (
                    <div key={phase} className={`bg-gray-900 rounded-lg p-3 ${
                      isServer ? 'opacity-40' :
                      isActive ? 'ring-1 ring-blue-500/50' : ''
                    }`}>
                      <div className="flex items-center justify-between mb-1.5">
                        <span className={`text-xs font-bold ${
                          isServer ? 'text-gray-600' :
                          isDone ? 'text-green-400' : isActive ? cfg.textColor : 'text-gray-600'
                        }`}>
                          {cfg.label}
                          {isServer && <span className="text-[8px] font-normal text-gray-600 ml-1">SVR</span>}
                        </span>
                        <div className="flex items-center gap-2">
                          {!isServer && (isDone || isActive) && fpm > 0 && (
                            <span className="text-[9px] font-mono text-yellow-400">
                              {isMcOnly ? `${(60 / fpm).toFixed(1)}s` : `${fpm.toFixed(1)}/m`}
                            </span>
                          )}
                          {!isServer && isDone && elapsed > 0 && (
                            <span className="text-[9px] font-mono text-gray-500">{elapsed.toFixed(1)}s</span>
                          )}
                          <span className={`text-[10px] font-mono ${
                            isServer ? 'text-gray-700' :
                            isDone ? 'text-green-400' : isActive ? 'text-gray-300' : 'text-gray-600'
                          }`}>
                            {isServer ? '-' :
                             isDone ? `${wp.phaseCount}/${wp.phaseCount}` :
                             isActive ? `${wp.phaseIndex}/${wp.phaseCount}` :
                             `-`}
                          </span>
                        </div>
                      </div>
                      <div className="w-full h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            isServer ? 'bg-gray-700' :
                            isDone ? 'bg-green-500' : isActive ? `${cfg.color} animate-pulse` : 'bg-gray-700'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Stats Grid — overall + per-phase speeds */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                <div className="bg-gray-900 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-gray-500 mb-1">{t('admin.queue_completed')}</div>
                  <div className="text-xl font-bold font-mono text-emerald-300">{completed}</div>
                </div>
                <div className="bg-gray-900 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-gray-500 mb-1">{t('admin.queue_pending')}</div>
                  <div className="text-xl font-bold font-mono text-yellow-300">{pending}</div>
                </div>
                <div className="bg-gray-900 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-gray-500 mb-1">{t('worker.throughput')}</div>
                  <div className="text-xl font-bold font-mono text-yellow-300">{perMin > 0 ? `${perMin.toFixed(1)}` : '-'}</div>
                  <div className="text-[9px] text-gray-600">/min</div>
                </div>
                <div className="bg-gray-900 rounded-lg p-3 text-center">
                  <div className="text-[10px] text-gray-500 mb-1">ETA</div>
                  <div className="text-xl font-bold font-mono text-emerald-300">{wp.etaMs > 0 ? formatEta(wp.etaMs) : '-'}</div>
                </div>
              </div>

              {/* Per-phase speed breakdown */}
              {(wp.phaseFpm?.parse > 0 || wp.phaseFpm?.vision > 0 || wp.phaseFpm?.embed_vv > 0 || wp.phaseFpm?.embed_mv > 0) && (
                <div className="flex items-center gap-3 mb-3 text-[10px] font-mono">
                  <span className="text-gray-500">{t('worker.phase_speed')}:</span>
                  {phaseOrder.map(phase => {
                    const fpm = wp.phaseFpm?.[phase] || 0;
                    if (fpm <= 0 || serverPhases.has(phase)) return null;
                    const cfg = phaseConfig[phase];
                    return (
                      <span key={phase} className={cfg.textColor}>
                        {cfg.label} {fpm.toFixed(1)}/m
                      </span>
                    );
                  })}
                </div>
              )}

              {/* Current file */}
              <div className="flex items-center gap-3 text-xs">
                {wp.currentPhase && (
                  <span className={`font-mono font-bold ${
                    serverPhases.has(wp.currentPhase) ? 'text-gray-600' :
                    (phaseConfig[wp.currentPhase]?.textColor || 'text-gray-400')
                  }`}>
                    {phaseConfig[wp.currentPhase]?.label || wp.currentPhase}
                  </span>
                )}
                <span className="text-gray-400 truncate">
                  {wp.currentFile?.split(/[/\\]/).pop() || t('worker.no_jobs')}
                </span>
              </div>
            </div>
          );
        })()}

        {/* Waiting for jobs */}
        {isWorkerRunning && !workerProgress.currentPhase && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 text-center text-gray-500 text-sm">
            {t('worker.no_jobs')}
          </div>
        )}

        {/* Log Panel */}
        <div className="bg-gray-800 rounded-lg border border-gray-700">
          <h3 className="text-sm font-medium text-gray-300 px-4 py-3 border-b border-gray-700">
            {t('worker.recent_log')}
          </h3>
          <div className="h-64 overflow-y-auto p-3 font-mono text-xs space-y-1">
            {logs.length === 0 ? (
              <div className="text-gray-600">{t('msg.no_logs')}</div>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="flex gap-2">
                  <span className="text-gray-600 flex-shrink-0">[{log.timestamp}]</span>
                  <span className={
                    log.type === 'error' ? 'text-red-400' :
                    log.type === 'success' ? 'text-green-400' :
                    log.type === 'warning' ? 'text-yellow-400' :
                    'text-gray-400'
                  }>
                    {log.message}
                  </span>
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </div>
        </div>

      </div>
    </div>
  );
}
