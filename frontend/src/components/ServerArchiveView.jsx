import React, { useState, useEffect, useCallback } from 'react';
import { ChevronDown, ChevronUp, RefreshCw, Layers, Users } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { MyWorkersSection } from '../pages/WorkerPage';
import FolderInfoBar from './FolderInfoBar';
import FileGrid from './FileGrid';

function QueueStatusBar({ appMode, reloadSignal }) {
  const { t } = useLocale();
  const [expanded, setExpanded] = useState(false);
  const [stats, setStats] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      // Server mode Electron: use IPC (bypass HTTP auth)
      if (isElectron && window.electron?.queue) {
        const data = await window.electron.queue.getStats();
        if (data.success !== false) setStats(data);
        return;
      }
      // Web mode: use HTTP API
      const data = await getJobStats();
      if (data.success !== false) setStats(data);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  // Refresh on reloadSignal (after queue registration)
  useEffect(() => {
    if (reloadSignal > 0) fetchStats();
  }, [reloadSignal, fetchStats]);

  const pending = stats?.pending ?? 0;
  const assigned = stats?.assigned ?? 0;
  const processing = stats?.processing ?? 0;
  const completed = stats?.completed ?? 0;
  const failed = stats?.failed ?? 0;
  const total = stats?.total ?? 0;

  const hasPending = pending > 0 || processing > 0 || assigned > 0;

  return (
    <div className="border-b border-gray-700 bg-gray-800/90 flex-shrink-0">
      {/* Always-visible compact status bar */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2 hover:bg-gray-700/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <Layers size={14} className="text-blue-400" />
            <span className="text-xs font-semibold text-gray-300">{t('archive.queue_dashboard')}</span>
          </div>

          {/* Inline stats chips */}
          {stats && (
            <div className="flex items-center gap-2">
              {pending > 0 && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-yellow-900/50 border border-yellow-700/50">
                  <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse" />
                  <span className="text-[11px] font-bold text-yellow-300">{pending}</span>
                  <span className="text-[10px] text-yellow-400/70">{t('admin.queue_pending')}</span>
                </span>
              )}
              {(assigned + processing) > 0 && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-cyan-900/50 border border-cyan-700/50">
                  <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
                  <span className="text-[11px] font-bold text-cyan-300">{assigned + processing}</span>
                  <span className="text-[10px] text-cyan-400/70">{t('admin.queue_processing')}</span>
                </span>
              )}
              {completed > 0 && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-green-900/40 border border-green-700/50">
                  <span className="text-[11px] font-bold text-green-300">{completed}</span>
                  <span className="text-[10px] text-green-400/70">{t('admin.queue_completed')}</span>
                </span>
              )}
              {failed > 0 && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-900/40 border border-red-700/50">
                  <span className="text-[11px] font-bold text-red-300">{failed}</span>
                  <span className="text-[10px] text-red-400/70">{t('admin.queue_failed')}</span>
                </span>
              )}
              {!hasPending && total === 0 && (
                <span className="text-[11px] text-gray-500">{t('archive.queue_no_pending')}</span>
              )}
            </div>
          )}
          {!stats && (
            <span className="text-[11px] text-gray-500 flex items-center gap-1">
              <RefreshCw size={10} className="animate-spin" />
              {t('status.loading')}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {stats && total > 0 && (
            <span className="text-[10px] text-gray-500 font-mono">{total} total</span>
          )}
          {expanded
            ? <ChevronUp size={14} className="text-gray-500" />
            : <ChevronDown size={14} className="text-gray-500" />
          }
        </div>
      </button>

      {/* Expanded: detailed grid + workers */}
      {expanded && (
        <div className="px-4 pb-3 space-y-3 border-t border-gray-700/50">
          {/* Queue Stats Grid */}
          {stats && (
            <div className="pt-2">
              <div className="grid grid-cols-6 gap-2">
                {[
                  { key: 'pending', value: pending, color: 'text-yellow-400' },
                  { key: 'assigned', value: assigned, color: 'text-blue-400' },
                  { key: 'processing', value: processing, color: 'text-cyan-400' },
                  { key: 'completed', value: completed, color: 'text-green-400' },
                  { key: 'failed', value: failed, color: 'text-red-400' },
                  { key: 'total', value: total, color: 'text-gray-300' },
                ].map(({ key, value, color }) => (
                  <div key={key} className="text-center">
                    <div className={`text-sm font-bold ${color}`}>{value}</div>
                    <div className="text-[10px] text-gray-500">{t(`admin.queue_${key}`)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Connected Workers (server mode only, Electron) */}
          {appMode === 'server' && <MyWorkersSection />}
        </div>
      )}
    </div>
  );
}

export default function ServerArchiveView({
  currentPath,
  selectedFiles,
  setSelectedFiles,
  selectedPaths,
  onProcessFolder,
  onFindSimilar,
  isProcessing,
  reloadSignal,
  appMode,
  queueReloadSignal,
}) {
  return (
    <div className="h-full flex flex-col">
      {/* Queue Status Bar - always visible at TOP */}
      <QueueStatusBar appMode={appMode} reloadSignal={queueReloadSignal} />

      <FolderInfoBar
        currentPath={currentPath}
        onProcessFolder={onProcessFolder}
        isProcessing={isProcessing}
        reloadSignal={reloadSignal}
        appMode={appMode}
      />
      <div className="flex-1 overflow-y-auto p-4 pb-16">
        <FileGrid
          currentPath={currentPath}
          selectedFiles={selectedFiles}
          setSelectedFiles={setSelectedFiles}
          selectedPaths={selectedPaths}
          onFindSimilar={onFindSimilar}
        />
      </div>
    </div>
  );
}
