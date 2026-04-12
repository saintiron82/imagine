import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Square, RefreshCw, Server, Activity, AlertCircle, Clock, Loader2, Cpu, Sliders, Zap, CalendarClock } from 'lucide-react';
import { useLocale } from '../i18n';
import { apiClient, isElectron, getServerUrl, getAccessToken, getRefreshToken } from '../api/client';
import { getJobStats } from '../api/worker';
import { listMyWorkers, stopMyWorker, getBenchmark, runBenchmark } from '../api/admin';

export function MyWorkersSection() {
  const { t } = useLocale();
  const [workers, setWorkers] = useState([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const data = await listMyWorkers();
      setWorkers(data.workers || []);
    } catch { /* ignore — user may not have workers */ }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, [load]);

  const handleStop = async (id) => {
    try {
      await stopMyWorker(id);
      load();
    } catch (e) {
      console.error('Failed to stop worker:', e);
    }
  };

  const timeAgo = (isoStr) => {
    if (!isoStr) return '-';
    const diff = Math.floor((Date.now() - new Date(isoStr).getTime()) / 1000);
    if (diff < 60) return t('worker.last_heartbeat_ago', { seconds: diff });
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    return `${Math.floor(diff / 3600)}h`;
  };

  const onlineWorkers = workers.filter(w => w.status === 'online');

  // Don't render if no workers at all
  if (!loading && workers.length === 0) return null;

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
        <Cpu size={14} className="text-green-400" />
        {t('worker.my_workers_title')}
        {onlineWorkers.length > 0 && (
          <span className="text-xs text-green-400 font-normal">({onlineWorkers.length} online)</span>
        )}
      </h3>
      {loading ? (
        <div className="text-sm text-gray-500">{t('status.loading')}</div>
      ) : onlineWorkers.length === 0 ? (
        <div className="text-sm text-gray-500">{t('worker.my_workers_empty')}</div>
      ) : (
        <div className="space-y-2">
          {onlineWorkers.map((w) => (
            <div key={w.id} className="flex items-center justify-between bg-gray-900 rounded px-3 py-2">
              <div className="flex items-center gap-3 min-w-0">
                <Activity size={14} className="text-green-400 animate-pulse flex-shrink-0" />
                <div className="min-w-0">
                  <div className="text-sm font-medium text-gray-200 truncate">{w.worker_name}</div>
                  <div className="text-xs text-gray-500">{timeAgo(w.last_heartbeat)}</div>
                </div>
                <span className="text-xs font-mono text-yellow-300 flex-shrink-0">B:{w.batch_capacity}</span>
                <span className="text-xs text-green-400 flex-shrink-0">{w.jobs_completed} done</span>
                {w.current_phase && (
                  <span className="text-xs text-blue-300 flex-shrink-0">{w.current_phase}</span>
                )}
              </div>
              <button
                onClick={() => handleStop(w.id)}
                className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-gray-700 hover:bg-red-900/50 text-gray-400 hover:text-red-300 flex-shrink-0"
              >
                <Square size={10} />
                {t('worker.action_stop_worker')}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


const DAY_KEYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'];
const DAY_ISO = [1, 2, 3, 4, 5, 6, 7]; // ISO weekday

function WorkerScheduleSettings({ t }) {
  const [autoProcessing, setAutoProcessing] = useState(true);
  const [restAfterBatch, setRestAfterBatch] = useState(30);
  const [workerRest, setWorkerRest] = useState(0);
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');
  const [activeDays, setActiveDays] = useState([1, 2, 3, 4, 5, 6, 7]);
  const [idleTimeout, setIdleTimeout] = useState(10);
  const [saved, setSaved] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!isElectron) { setLoaded(true); return; }
    const load = async () => {
      try {
        const result = await window.electron?.pipeline?.getConfig();
        if (result?.success) {
          const w = result.config?.worker || {};
          const s = result.config?.server || {};
          if (w.rest_after_batch_s != null) setWorkerRest(w.rest_after_batch_s);
          if (w.idle_unload_minutes != null) setIdleTimeout(w.idle_unload_minutes);
          // Auto processing settings (server config)
          if (s.auto_processing) {
            if (s.auto_processing.enabled != null) setAutoProcessing(s.auto_processing.enabled);
            if (s.auto_processing.rest_after_batch_s != null) setRestAfterBatch(s.auto_processing.rest_after_batch_s);
          }
          if (w.schedule) {
            const sched = w.schedule;
            if (sched.active_hours) {
              const parts = sched.active_hours.split('-');
              if (parts.length === 2) {
                setStartTime(parts[0].trim());
                setEndTime(parts[1].trim());
              }
            }
            if (sched.active_days && Array.isArray(sched.active_days)) {
              setActiveDays(sched.active_days);
            }
          }
        }
      } catch (e) { console.error('Failed to load schedule config:', e); }
      setLoaded(true);
    };
    load();
  }, []);

  const saveSetting = useCallback(async (key, value) => {
    if (!isElectron) return;
    try {
      await window.electron.pipeline.updateConfig(key, value);
      setSaved(true);
      setTimeout(() => setSaved(false), 1500);
    } catch (e) { console.error('Failed to save schedule setting:', e); }
  }, []);

  const toggleDay = (day) => {
    const newDays = activeDays.includes(day)
      ? activeDays.filter(d => d !== day)
      : [...activeDays, day].sort();
    // All days selected = empty array (every day)
    const toSave = newDays.length === 7 ? [] : newDays;
    setActiveDays(newDays);
    saveSetting('worker.schedule.active_days', toSave);
  };

  const updateHours = (start, end) => {
    if (!start && !end) {
      saveSetting('worker.schedule.active_hours', '');
    } else if (start && end) {
      saveSetting('worker.schedule.active_hours', `${start}-${end}`);
    }
  };

  if (!loaded) return null;

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <h3 className="text-sm font-medium text-gray-300 mb-1 flex items-center gap-2">
        <CalendarClock size={14} className="text-cyan-400" />
        {t('worker.schedule_title')}
        {saved && <span className="text-xs text-green-400 ml-auto">{t('worker.schedule_saved')}</span>}
      </h3>
      <p className="text-xs text-gray-500 mb-4">{t('worker.schedule_desc')}</p>

      <div className="space-y-5">
        {/* Worker Rest After Batch */}
        <div>
          <label className="text-xs text-gray-400 mb-2 block">{t('worker.worker_rest_title')}</label>
          <div className="flex items-center gap-2">
            <input
              type="number" min="0" max="300" value={workerRest}
              onChange={e => {
                const v = Math.max(0, Math.min(300, parseInt(e.target.value) || 0));
                setWorkerRest(v);
                saveSetting('worker.rest_after_batch_s', v);
              }}
              className="w-20 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white"
            />
            <span className="text-xs text-gray-500">{t('worker.rest_unit')}</span>
          </div>
          <div className="text-[10px] text-gray-500 mt-1">
            {t('worker.worker_rest_desc')}
          </div>
        </div>

        {/* Auto Processing */}
        <div>
          <label className="text-xs text-gray-400 mb-2 block">{t('worker.auto_title')}</label>
          <div className="flex items-center gap-3">
            <button
              onClick={async () => {
                const newVal = !autoProcessing;
                setAutoProcessing(newVal);
                // Use PATCH API to also start/stop embedded worker
                try {
                  const { apiClient } = await import('../api/client');
                  await apiClient.patch('/api/v1/admin/workers/auto-processing', { enabled: newVal });
                } catch { /* fallback */
                  saveSetting('server.auto_processing.enabled', newVal);
                }
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
          <div className="text-[10px] text-gray-500 mt-1">
            {t('worker.auto_desc')}
          </div>
        </div>

        {/* Rest After Batch */}
        {autoProcessing && (
          <div>
            <label className="text-xs text-gray-400 mb-2 block">{t('worker.rest_title')}</label>
            <div className="flex items-center gap-2">
              <input
                type="number" min="0" max="300" value={restAfterBatch}
                onChange={e => {
                  const v = parseInt(e.target.value) || 0;
                  setRestAfterBatch(v);
                  saveSetting('server.auto_processing.rest_after_batch_s', v);
                }}
                className="w-20 bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs text-white"
              />
              <span className="text-xs text-gray-500">{t('worker.rest_unit')}</span>
            </div>
            <div className="text-[10px] text-gray-500 mt-1">
              {t('worker.rest_desc')}
            </div>
          </div>
        )}

        {/* Active Hours */}
        <div>
          <label className="text-xs text-gray-400 mb-2 block">{t('worker.schedule_active_hours')}</label>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-gray-500">{t('worker.schedule_start')}</span>
              <input
                type="time"
                value={startTime}
                onChange={(e) => {
                  setStartTime(e.target.value);
                  updateHours(e.target.value, endTime);
                }}
                className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-xs text-gray-300 focus:border-cyan-500 focus:outline-none"
              />
            </div>
            <span className="text-gray-600">—</span>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-gray-500">{t('worker.schedule_end')}</span>
              <input
                type="time"
                value={endTime}
                onChange={(e) => {
                  setEndTime(e.target.value);
                  updateHours(startTime, e.target.value);
                }}
                className="bg-gray-900 border border-gray-600 rounded px-2 py-1 text-xs text-gray-300 focus:border-cyan-500 focus:outline-none"
              />
            </div>
            {(startTime || endTime) && (
              <button
                onClick={() => {
                  setStartTime('');
                  setEndTime('');
                  updateHours('', '');
                }}
                className="text-xs text-gray-500 hover:text-gray-300"
              >
                ×
              </button>
            )}
          </div>
          <div className="text-[10px] text-gray-600 mt-1">{t('worker.schedule_hours_desc')}</div>
        </div>

        {/* Active Days */}
        <div>
          <label className="text-xs text-gray-400 mb-2 block">{t('worker.schedule_active_days')}</label>
          <div className="flex gap-1.5">
            {DAY_KEYS.map((key, i) => {
              const iso = DAY_ISO[i];
              const active = activeDays.includes(iso);
              return (
                <button
                  key={key}
                  onClick={() => toggleDay(iso)}
                  className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                    active
                      ? 'bg-cyan-600 text-white'
                      : 'bg-gray-700 text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {t(`worker.schedule_day_${key}`)}
                </button>
              );
            })}
          </div>
        </div>

        {/* Idle Unload Timeout */}
        <div>
          <div className="flex justify-between items-center mb-1">
            <label className="text-xs text-gray-400">{t('worker.schedule_idle_timeout')}</label>
            <span className="text-xs font-mono text-cyan-300">
              {idleTimeout <= 0 ? t('worker.schedule_idle_off') : t('worker.schedule_idle_value', { minutes: idleTimeout })}
            </span>
          </div>
          <input
            type="range" min="0" max="60" step="5" value={idleTimeout}
            onChange={(e) => {
              const v = Number(e.target.value);
              setIdleTimeout(v);
              saveSetting('worker.idle_unload_minutes', v);
            }}
            className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
          />
          <div className="flex justify-between text-[10px] text-gray-600 mt-0.5">
            <span>{t('worker.schedule_idle_off')}</span>
            <span className="text-gray-500">{t('worker.schedule_idle_desc')}</span>
            <span>60</span>
          </div>
        </div>
      </div>
    </div>
  );
}


function WorkerPage({ appMode }) {
  const { t } = useLocale();
  const [stats, setStats] = useState(null);
  const [workerStatus, setWorkerStatus] = useState('idle'); // idle | running | stopping | error
  const [logs, setLogs] = useState([]);
  const [currentJobs, setCurrentJobs] = useState([]);
  const [processingMode, setProcessingMode] = useState(null); // mc/vv/mv/parse/idle
  const [currentFile, setCurrentFile] = useState('');
  const [batchProgress, setBatchProgress] = useState(null); // {index, count, phase}
  const startTimeRef = useRef(null);
  // Phase-level stats (persist throughout session)
  const [phaseStats, setPhaseStats] = useState({
    mc:  { completed: 0, failed: 0, startTime: null },
    vv:  { completed: 0, failed: 0, startTime: null },
    mv:  { completed: 0, failed: 0, startTime: null },
    parse: { completed: 0, failed: 0, startTime: null },
  });
  const recordPhaseResult = useCallback((phase, success) => {
    setPhaseStats(prev => {
      const p = prev[phase] || { completed: 0, failed: 0, startTime: null };
      return {
        ...prev,
        [phase]: {
          completed: p.completed + (success ? 1 : 0),
          failed: p.failed + (success ? 0 : 1),
          startTime: p.startTime || Date.now(),
        }
      };
    });
  }, []);
  // logEndRef removed — using logContainerRef for scroll
  const pollRef = useRef(null);

  const addLog = useCallback((message, type = 'info') => {
    const entry = { message, type, timestamp: new Date().toLocaleTimeString() };
    setLogs(prev => {
      const next = [...prev, entry];
      return next.length > 100 ? next.slice(-100) : next;
    });
  }, []);

  // Poll stats + worker status every 5 seconds
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const data = await getJobStats();
        if (data.success !== false) setStats(data);
      } catch { /* ignore */ }
    };

    const fetchWorkerStatus = async () => {
      if (isElectron && window.electron?.worker) return; // IPC mode handles its own status via events
      try {
        const data = await apiClient.get('/api/v1/admin/worker/status');
        setWorkerStatus(data.running ? 'running' : 'idle');
        if (data.last_error) {
          setWorkerStatus('error');
        }
      } catch { /* ignore — user may not be admin */ }
    };

    fetchStats();
    fetchWorkerStatus();
    pollRef.current = setInterval(() => {
      fetchStats();
      fetchWorkerStatus();
    }, 5000);
    return () => clearInterval(pollRef.current);
  }, []);

  // Auto-scroll logs — only within the log container, not the whole page
  const logContainerRef = useRef(null);
  useEffect(() => {
    const container = logContainerRef.current;
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
  }, [logs]);

  // Worker IPC listeners (Electron only)
  useEffect(() => {
    if (!isElectron) return;
    const w = window.electron?.worker;
    if (!w) return;

    const onStatus = (data) => {
      setWorkerStatus(data.status);
      if (data.jobs) setCurrentJobs(data.jobs);
    };
    const onLog = (data) => addLog(data.message, data.type);
    const onJobDone = (data) => {
      // Phase from current processingMode (job_done doesn't carry phase)
      const phase = processingMode || 'mc';
      recordPhaseResult(phase, data.success !== false);
      setCurrentFile('');
    };
    const onBatchFileDone = (data) => {
      setCurrentFile(data.file_name || '');
      setBatchProgress({ index: data.index, count: data.count, phase: data.phase });
      // Map batch phase names to standard mode names
      const phaseMap = { 'vision': 'mc', 'embed_vv': 'vv', 'embed_mv': 'mv', 'parse': 'parse' };
      const phase = phaseMap[data.phase] || data.phase || processingMode || 'mc';
      recordPhaseResult(phase, !!data.success);
    };
    const onMode = (data) => {
      setProcessingMode(data.mode);
    };

    w.onStatus?.(onStatus);
    w.onLog?.(onLog);
    w.onJobDone?.(onJobDone);
    w.onBatchFileDone?.(onBatchFileDone);
    w.onProcessingMode?.(onMode);

    return () => {
      w.offStatus?.();
      w.offLog?.();
      w.offJobDone?.();
      w.offBatchFileDone?.();
      w.offProcessingMode?.();
    };
  }, [addLog, recordPhaseResult, processingMode]);

  const handleStart = async () => {
    if (isElectron && window.electron?.worker?.start) {
      // Electron: spawn worker_ipc.py via IPC
      const serverUrl = getServerUrl();
      const accessToken = getAccessToken();
      const refreshToken = getRefreshToken();

      if (!serverUrl || !accessToken) {
        addLog('Not logged in. Please login first.', 'error');
        return;
      }

      try {
        const result = await window.electron.worker.start({
          serverUrl,
          accessToken,
          refreshToken: refreshToken || '',
        });
        if (result.success === false) {
          addLog(result.error || 'Failed to start worker', 'error');
          return;
        }
        setWorkerStatus('running');
        setPhaseStats({
          mc:  { completed: 0, failed: 0, startTime: null },
          vv:  { completed: 0, failed: 0, startTime: null },
          mv:  { completed: 0, failed: 0, startTime: null },
          parse: { completed: 0, failed: 0, startTime: null },
        });
        startTimeRef.current = Date.now();
        addLog(t('worker.connecting'), 'info');
      } catch (e) {
        addLog(e.message, 'error');
        setWorkerStatus('error');
      }
    } else {
      // Web: server embedded worker API
      try {
        const result = await apiClient.post('/api/v1/admin/worker/start');
        if (result.success === false) {
          addLog(result.error || 'Failed to start worker', 'error');
          return;
        }
        setWorkerStatus('running');
        addLog(t('worker.connecting'), 'info');
      } catch (e) {
        addLog(e.detail || e.message, 'error');
        setWorkerStatus('error');
      }
    }
  };

  const handleStop = async () => {
    if (isElectron && window.electron?.worker?.stop) {
      try {
        await window.electron.worker.stop();
        setWorkerStatus('idle');
        setCurrentJobs([]);
        addLog(t('worker.stop'), 'info');
      } catch (e) {
        addLog(e.message, 'error');
      }
    } else {
      setWorkerStatus('stopping');
      try {
        const result = await apiClient.post('/api/v1/admin/worker/stop');
        setWorkerStatus('idle');
        setCurrentJobs([]);
        addLog(t('worker.stop'), 'info');
        if (result.jobs_completed) {
          addLog(`${t('worker.jobs_completed')}: ${result.jobs_completed}`, 'success');
        }
      } catch (e) {
        addLog(e.detail || e.message, 'error');
        setWorkerStatus('error');
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

  return (
    <div className="h-full overflow-y-auto p-6 space-y-6">
      <div className="max-w-4xl mx-auto space-y-6">

        {/* Header — mode-specific */}
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            {appMode === 'server' ? (
              <><Server size={22} className="text-blue-400" />{t('worker.server_dashboard')}</>
            ) : (
              <><Zap size={22} className="text-emerald-400" />{t('worker.client_worker')}</>
            )}
          </h2>
          {appMode === 'client' && getServerUrl() && (
            <p className="text-xs text-gray-500 mt-1">{t('worker.connected_to', { url: getServerUrl() })}</p>
          )}
        </div>

        {/* Worker Control (Electron only — web has no processing capability) */}
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

        {/* Web mode: no processing capability notice */}
        {!isElectron && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 text-center">
            <p className="text-sm text-gray-400">{t('worker.web_mode_notice')}</p>
          </div>
        )}

        {/* Performance Limits (Electron only) */}
        {/* Performance limits removed — scheduler handles batch sizing */}

        {/* Schedule Settings (Electron only) */}
        {isElectron && <WorkerScheduleSettings t={t} />}

        {/* My Workers (server mode only) */}
        {appMode === 'server' && <MyWorkersSection />}

        {/* My Worker Status — all phases with active highlight */}
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Activity size={14} />
              {t('worker.my_status')}
            </h3>
            {startTimeRef.current && (
              <span className="text-[11px] text-gray-500">
                {Math.floor((Date.now() - startTimeRef.current) / 60000)}m {t('worker.uptime')}
              </span>
            )}
          </div>

          {/* Phase cards — all phases shown, active one highlighted */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {[
              { id: 'parse', label: 'Parse', color: 'sky', icon: '📄' },
              { id: 'mc', label: 'MC', color: 'purple', icon: '🧠' },
              { id: 'vv', label: 'VV', color: 'blue', icon: '👁' },
              { id: 'mv', label: 'MV', color: 'green', icon: '📝' },
            ].map(phase => {
              const isActive = processingMode === phase.id;
              const ps = phaseStats[phase.id] || { completed: 0, failed: 0, startTime: null };
              const hasWork = ps.completed > 0 || ps.failed > 0;
              const fpm = ps.startTime && ps.completed > 0
                ? (ps.completed / ((Date.now() - ps.startTime) / 60000)).toFixed(1)
                : null;
              return (
                <div
                  key={phase.id}
                  className={`rounded-lg p-3 border transition-all ${
                    isActive
                      ? `border-${phase.color}-500/60 bg-${phase.color}-500/10 ring-1 ring-${phase.color}-500/30`
                      : hasWork
                        ? `border-${phase.color}-500/30 bg-${phase.color}-500/5`
                        : 'border-gray-700/50 bg-gray-900/30 opacity-40'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`text-xs font-bold ${isActive || hasWork ? `text-${phase.color}-300` : 'text-gray-500'}`}>
                      {phase.icon} {phase.label}
                    </span>
                    {isActive && (
                      <span className={`w-2 h-2 rounded-full bg-${phase.color}-400 animate-pulse`} />
                    )}
                  </div>
                  {hasWork ? (
                    <div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-lg font-bold text-white">{ps.completed}</span>
                        {ps.failed > 0 && (
                          <span className="text-xs text-red-400">+{ps.failed} fail</span>
                        )}
                      </div>
                      <div className="text-[11px] text-gray-400 mt-0.5">
                        {fpm ? `${fpm} files/min` : isActive ? 'starting...' : ''}
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-gray-600">{isActive ? 'starting...' : '\u2014'}</div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Current file progress */}
          {currentFile && (
            <div className="mt-3 flex items-center gap-2 px-1">
              <Loader2 size={12} className="animate-spin text-blue-400 shrink-0" />
              <span className="text-xs text-gray-400 truncate">
                {batchProgress ? `[${batchProgress.index}/${batchProgress.count}] ` : ''}
                {currentFile}
              </span>
            </div>
          )}
        </div>

        {/* Current Jobs */}
        {currentJobs.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 className="text-sm font-medium text-gray-300 mb-3">{t('worker.current_jobs')}</h3>
            <div className="space-y-2">
              {currentJobs.map((job, i) => (
                <div key={job.id || i} className="flex items-center gap-3 text-sm">
                  <Loader2 size={14} className="animate-spin text-blue-400" />
                  <span className="text-gray-300 truncate flex-1">{job.file_path?.split('/').pop()}</span>
                  <span className="text-xs text-gray-500">{job.phase || ''}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {currentJobs.length === 0 && workerStatus === 'running' && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 text-center text-gray-500 text-sm">
            {t('worker.no_jobs')}
          </div>
        )}

        {/* Log Panel */}
        <div className="bg-gray-800 rounded-lg border border-gray-700">
          <h3 className="text-sm font-medium text-gray-300 px-4 py-3 border-b border-gray-700">
            {t('worker.recent_log')}
          </h3>
          <div ref={logContainerRef} className="h-64 overflow-y-auto p-3 font-mono text-xs space-y-1">
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
          </div>
        </div>

        {/* Benchmark */}
        <BenchmarkSection />

      </div>
    </div>
  );
}

export function BenchmarkSection() {
  const { t } = useLocale();
  const [data, setData] = useState(null);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const result = await getBenchmark();
      setData(result);
    } catch { /* ignore */ }
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleRun = async () => {
    setRunning(true);
    try {
      const result = await runBenchmark();
      setData(result);
    } catch (e) {
      console.error('Benchmark failed:', e);
    }
    setRunning(false);
  };

  const gradeColor = (grade) => ({
    S: 'text-yellow-400', A: 'text-green-400', B: 'text-blue-400',
    C: 'text-orange-400', F: 'text-red-400',
  }[grade] || 'text-gray-400');

  const scores = data?.scores;
  const hw = data?.hardware;
  const profile = data?.profile;

  return (
    <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-lg font-semibold">{t('worker.bench_title')}</h3>
          <p className="text-sm text-gray-400 mt-1">{t('worker.bench_desc')}</p>
        </div>
        <button
          onClick={handleRun}
          disabled={running}
          className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
            running
              ? 'bg-gray-600 text-gray-400 cursor-wait'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          }`}
        >
          {running ? (
            <span className="flex items-center gap-2">
              <Loader2 size={14} className="animate-spin" />
              {t('worker.bench_running')}
            </span>
          ) : t('worker.bench_run')}
        </button>
      </div>

      {/* Hardware Info */}
      {hw && (
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm mb-4 p-3 bg-gray-900/50 rounded">
          <div className="flex justify-between">
            <span className="text-gray-500">CPU</span>
            <span className="text-gray-300 font-mono">{hw.cpu_name || '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">코어</span>
            <span className="text-gray-300 font-mono">{hw.cpu_count || '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">메모리</span>
            <span className="text-gray-300 font-mono">{hw.memory_total_gb ? `${hw.memory_total_gb} GB` : '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">GPU</span>
            <span className="text-gray-300 font-mono">{hw.gpu_name || 'CPU'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">VRAM</span>
            <span className="text-gray-300 font-mono">{hw.vram_gb ? `${hw.vram_gb} GB` : '—'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">OS</span>
            <span className="text-gray-300 font-mono">{hw.os || '—'} {hw.arch || ''}</span>
          </div>
        </div>
      )}

      {scores ? (
        <div className="space-y-3">
          {/* Grade + Total */}
          <div className="flex items-center gap-4 p-3 bg-gray-900/50 rounded">
            <span className={`text-3xl font-bold ${gradeColor(scores.grade)}`}>
              {scores.grade}
            </span>
            <div className="flex-1">
              <div className="text-sm text-gray-400">{t('worker.bench_score')}</div>
              <div className="text-xl font-mono">{scores.total?.toLocaleString()}</div>
            </div>
          </div>

          {/* Incapable warning */}
          {scores.incapable?.length > 0 && (
            <div className="px-3 py-2 bg-red-900/30 border border-red-800 rounded text-sm text-red-300">
              {scores.incapable.map(p => p.toUpperCase()).join(', ')} 처리 불가
            </div>
          )}

          {/* Per-phase */}
          <div className="grid grid-cols-3 gap-3">
            {['mc', 'vv', 'mv'].map(phase => {
              const p = scores.phases?.[phase] || {};
              return (
                <div key={phase} className={`p-3 rounded text-center ${
                  p.status === 'incapable' ? 'bg-red-900/20 border border-red-800/50' : 'bg-gray-900/50'
                }`}>
                  <div className="text-xs text-gray-500 uppercase mb-1">
                    {phase.toUpperCase()}
                  </div>
                  {p.status === 'incapable' ? (
                    <div className="text-lg font-bold text-red-400">불가</div>
                  ) : p.status === 'unmeasured' ? (
                    <div className="text-lg text-gray-500">미측정</div>
                  ) : (
                    <>
                      <div className="text-lg font-mono font-bold text-gray-200">
                        {p.speed}<span className="text-xs text-gray-500">/m</span>
                      </div>
                      <div className="text-sm font-mono text-emerald-400">
                        {p.weighted?.toLocaleString()}
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>

          {profile?.benchmarked_at && (
            <div className="text-xs text-gray-500 text-right">
              {t('worker.bench_last')}: {profile.benchmarked_at}
            </div>
          )}
        </div>
      ) : !loading && (
        <div className="text-center text-gray-500 py-6">
          {t('worker.bench_no_data')}
        </div>
      )}
    </div>
  );
}

export default WorkerPage;
