import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Square, RefreshCw, Server, Activity, AlertCircle, Clock, CheckCircle2, XCircle, Loader2 } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron, getServerUrl, getAccessToken, getRefreshToken } from '../api/client';
import { getJobStats } from '../api/worker';

function WorkerPage() {
  const { t } = useLocale();
  const [stats, setStats] = useState(null);
  const [workerStatus, setWorkerStatus] = useState('idle'); // idle | running | error
  const [logs, setLogs] = useState([]);
  const [currentJobs, setCurrentJobs] = useState([]);
  const logEndRef = useRef(null);
  const pollRef = useRef(null);

  const addLog = useCallback((message, type = 'info') => {
    const entry = { message, type, timestamp: new Date().toLocaleTimeString() };
    setLogs(prev => {
      const next = [...prev, entry];
      return next.length > 100 ? next.slice(-100) : next;
    });
  }, []);

  // Poll stats every 5 seconds
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
    if (!isElectron) return;
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
  };

  const handleStop = async () => {
    if (!isElectron) return;
    try {
      await window.electron.worker.stop();
      setWorkerStatus('idle');
      setCurrentJobs([]);
      addLog(t('worker.stop'), 'info');
    } catch (e) {
      addLog(e.message, 'error');
    }
  };

  const statusColor = {
    idle: 'text-gray-400',
    running: 'text-green-400',
    error: 'text-red-400',
  };

  const statusIcon = {
    idle: <Clock size={16} />,
    running: <Activity size={16} className="animate-pulse" />,
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

        {/* Web mode notice */}
        {!isElectron && (
          <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 text-yellow-300 text-sm">
            <AlertCircle size={16} className="inline mr-2" />
            {t('worker.web_mode_notice')}
          </div>
        )}

        {/* Worker Control */}
        {isElectron && (
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className={`flex items-center gap-2 ${statusColor[workerStatus]}`}>
                  {statusIcon[workerStatus]}
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
                  disabled={workerStatus === 'running' || !getAccessToken()}
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
