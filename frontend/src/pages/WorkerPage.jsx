import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Square, RefreshCw, Server, Activity, AlertCircle, Clock, CheckCircle2, XCircle, Loader2, Download, Copy, CheckCircle, Monitor, Cpu } from 'lucide-react';
import { useLocale } from '../i18n';
import { apiClient, isElectron, getServerUrl, getAccessToken, getRefreshToken } from '../api/client';
import { getJobStats } from '../api/worker';
import { registerWorker, listMyWorkers, stopMyWorker } from '../api/admin';

function MyWorkersSection() {
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


function ConnectMyPC() {
  const { t } = useLocale();
  const [downloading, setDownloading] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const data = await registerWorker();
      if (!data.token) throw new Error('No token returned');

      const serverUrl = window.location.origin;
      const scriptUrl = `/api/v1/worker/setup-script?token=${encodeURIComponent(data.token)}&server_url=${encodeURIComponent(serverUrl)}`;

      // Trigger file download
      const a = document.createElement('a');
      a.href = scriptUrl;
      a.download = 'setup_worker.py';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      setDownloaded(true);
    } catch (e) {
      console.error('Failed to register worker:', e);
    }
    setDownloading(false);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText('python3 setup_worker.py');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <h3 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
        <Monitor size={14} className="text-blue-400" />
        {t('worker.connect_title')}
      </h3>
      <p className="text-xs text-gray-500 mb-3">{t('worker.connect_desc')}</p>

      <div className="flex items-center gap-3 flex-wrap">
        <button
          onClick={handleDownload}
          disabled={downloading}
          className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 text-white rounded text-sm font-medium transition-colors"
        >
          {downloading ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
          {t('worker.download_setup')}
        </button>

        {downloaded && (
          <span className="text-xs text-green-400">{t('worker.setup_downloaded')}</span>
        )}
      </div>

      {downloaded && (
        <div className="mt-3 space-y-2">
          <div className="text-xs text-gray-400">{t('worker.run_instruction')}</div>
          <div className="flex gap-2">
            <code className="flex-1 px-3 py-2 bg-gray-900 rounded text-xs font-mono text-blue-300 select-all">
              python3 setup_worker.py
            </code>
            <button
              onClick={handleCopy}
              className="flex items-center gap-1 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-xs text-white flex-shrink-0"
            >
              {copied ? <CheckCircle size={12} className="text-green-400" /> : <Copy size={12} />}
            </button>
          </div>
          <p className="text-xs text-gray-600">{t('worker.python_required')}</p>
        </div>
      )}
    </div>
  );
}


function WorkerPage() {
  const { t } = useLocale();
  const [stats, setStats] = useState(null);
  const [workerStatus, setWorkerStatus] = useState('idle'); // idle | running | stopping | error
  const [logs, setLogs] = useState([]);
  const [currentJobs, setCurrentJobs] = useState([]);
  const [jobsCompleted, setJobsCompleted] = useState(0);
  const logEndRef = useRef(null);
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
      if (isElectron) return;
      try {
        const data = await apiClient.get('/api/v1/admin/worker/status');
        setWorkerStatus(data.running ? 'running' : 'idle');
        setJobsCompleted(data.jobs_completed || 0);
        if (data.last_error) {
          setWorkerStatus('error');
        }
      } catch { /* ignore - user may not be admin */ }
    };

    fetchStats();
    fetchWorkerStatus();
    pollRef.current = setInterval(() => {
      fetchStats();
      fetchWorkerStatus();
    }, 5000);
    return () => clearInterval(pollRef.current);
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
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
      addLog(`Completed: ${data.file_name || data.file_path}`, 'success');
    };

    w.onStatus?.(onStatus);
    w.onLog?.(onLog);
    w.onJobDone?.(onJobDone);

    return () => {
      w.offStatus?.();
      w.offLog?.();
      w.offJobDone?.();
    };
  }, [addLog]);

  const handleStart = async () => {
    if (isElectron) {
      // Electron mode: use IPC
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
        addLog(t('worker.connecting'), 'info');
      } catch (e) {
        addLog(e.message, 'error');
        setWorkerStatus('error');
      }
    } else {
      // Web mode: call server embedded worker API
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
    if (isElectron) {
      try {
        await window.electron.worker.stop();
        setWorkerStatus('idle');
        setCurrentJobs([]);
        addLog(t('worker.stop'), 'info');
      } catch (e) {
        addLog(e.message, 'error');
      }
    } else {
      // Web mode: call server embedded worker stop API
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

        {/* Header */}
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <Server size={22} className="text-blue-400" />
            {t('worker.title')}
          </h2>
        </div>

        {/* Worker Control */}
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className={`flex items-center gap-2 ${statusColor[workerStatus] || 'text-gray-400'}`}>
                {statusIcon[workerStatus] || <Clock size={16} />}
                {t(`worker.status_${workerStatus}`)}
              </span>
              {!isElectron && workerStatus === 'running' && jobsCompleted > 0 && (
                <span className="text-xs text-green-400 font-mono">
                  {t('worker.jobs_completed')}: {jobsCompleted}
                </span>
              )}
              {isElectron && workerStatus !== 'running' && getServerUrl() && (
                <span className="text-xs text-gray-500 truncate max-w-[200px]">
                  {getServerUrl()}
                </span>
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleStart}
                disabled={workerStatus === 'running' || workerStatus === 'stopping' || (isElectron && !getAccessToken())}
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

        {/* My Workers */}
        <MyWorkersSection />

        {/* Connect My PC */}
        <ConnectMyPC />

        {/* Queue Stats */}
        <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
          <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
            <RefreshCw size={14} className={stats ? '' : 'animate-spin'} />
            {t('worker.queue_stats')}
          </h3>
          {stats ? (
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
              {[
                { key: 'pending', value: stats.pending, color: 'text-yellow-400' },
                { key: 'assigned', value: stats.assigned, color: 'text-blue-400' },
                { key: 'processing', value: stats.processing, color: 'text-cyan-400' },
                { key: 'completed', value: stats.completed, color: 'text-green-400' },
                { key: 'failed', value: stats.failed, color: 'text-red-400' },
                { key: 'total', value: stats.total, color: 'text-gray-300' },
              ].map(({ key, value, color }) => (
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

export default WorkerPage;
