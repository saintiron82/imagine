import React, { useState, useEffect, useCallback } from 'react';
import { ChevronDown, ChevronUp, RefreshCw, Layers } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { MyWorkersSection } from '../pages/WorkerPage';
import FolderInfoBar from './FolderInfoBar';
import FileGrid from './FileGrid';

function QueueDashboard({ appMode }) {
  const { t } = useLocale();
  const [expanded, setExpanded] = useState(false);
  const [stats, setStats] = useState(null);

  const fetchStats = useCallback(async () => {
    try {
      const data = await getJobStats();
      if (data.success !== false) setStats(data);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  const statItems = [
    { key: 'pending', value: stats?.pending, color: 'text-yellow-400' },
    { key: 'assigned', value: stats?.assigned, color: 'text-blue-400' },
    { key: 'processing', value: stats?.processing, color: 'text-cyan-400' },
    { key: 'completed', value: stats?.completed, color: 'text-green-400' },
    { key: 'failed', value: stats?.failed, color: 'text-red-400' },
    { key: 'total', value: stats?.total, color: 'text-gray-300' },
  ];

  const summaryText = stats
    ? `${stats.pending ?? 0} pending / ${stats.processing ?? 0} processing / ${stats.completed ?? 0} done`
    : t('status.loading');

  return (
    <div className="border-t border-gray-700 bg-gray-800/80">
      {/* Toggle header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2 hover:bg-gray-700/50 transition-colors text-xs"
      >
        <div className="flex items-center gap-2 text-gray-400">
          <Layers size={12} />
          <span className="font-medium">{t('archive.queue_dashboard')}</span>
          {!expanded && stats && (
            <span className="text-gray-500 ml-2">{summaryText}</span>
          )}
        </div>
        {expanded ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronUp size={14} className="text-gray-500" />}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-3 space-y-3">
          {/* Queue Stats */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <RefreshCw size={12} className={stats ? 'text-gray-500' : 'animate-spin text-gray-500'} />
              <span className="text-xs font-medium text-gray-400">{t('worker.queue_stats')}</span>
            </div>
            {stats ? (
              <div className="grid grid-cols-6 gap-2">
                {statItems.map(({ key, value, color }) => (
                  <div key={key} className="text-center">
                    <div className={`text-sm font-bold ${color}`}>{value ?? 0}</div>
                    <div className="text-[10px] text-gray-500">{t(`admin.queue_${key}`)}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-gray-500">{t('status.loading')}</div>
            )}
          </div>

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
}) {
  return (
    <div className="h-full flex flex-col">
      <FolderInfoBar
        currentPath={currentPath}
        onProcessFolder={onProcessFolder}
        isProcessing={isProcessing}
        reloadSignal={reloadSignal}
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
      <QueueDashboard appMode={appMode} />
    </div>
  );
}
