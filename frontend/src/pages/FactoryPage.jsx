/**
 * FactoryPage — Processing factory: Queue dashboard + Workers + Dashboard + Logs
 * Combines panels previously in AdminPage + ClientWorkerView
 */

import { useState, useEffect, useCallback } from 'react';
import { Activity, Server, BarChart3, FileText, RefreshCw, Monitor } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { WorkersPanel, DashboardPanel } from './AdminPage';
import { MyWorkersSection, ConnectMyPC } from './WorkerPage';
import ClientWorkerView from '../components/ClientWorkerView';
import PipelineBlackboard from '../components/PipelineBlackboard';

export default function FactoryPage({
  isAdmin,
  appMode,
  isWorkerRunning,
  workerProgress,
  onWorkerStart,
  onWorkerStop,
  queueReloadSignal,
}) {
  const { t } = useLocale();
  const [activeTab, setActiveTab] = useState('pipeline');
  const [stats, setStats] = useState(null);

  // Fetch queue stats for summary bar
  const fetchStats = useCallback(async () => {
    try {
      if (isElectron && window.electron?.queue) {
        const data = await window.electron.queue.getStats();
        if (data.success !== false) setStats(data);
        return;
      }
      const data = await getJobStats();
      if (data.success !== false) setStats(data);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, [fetchStats]);

  const pending = stats?.pending ?? 0;
  const processing = (stats?.assigned ?? 0) + (stats?.processing ?? 0);
  const completed = stats?.completed ?? 0;
  const failed = stats?.failed ?? 0;

  const tabs = [
    { id: 'pipeline', label: t('factory.tab_pipeline'), icon: Monitor },
    { id: 'workers', label: t('factory.tab_workers'), icon: Server },
    { id: 'dashboard', label: t('factory.tab_dashboard'), icon: BarChart3 },
    { id: 'logs', label: t('factory.tab_logs'), icon: FileText },
  ];

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      {/* Summary bar with live stats */}
      <div className="px-4 py-2.5 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-4 text-sm">
          <span className="text-gray-400">
            {t('factory.summary_pending')}:{' '}
            <span className={`font-bold ${pending > 0 ? 'text-yellow-400' : 'text-gray-500'}`}>{pending}</span>
          </span>
          <span className="text-gray-400">
            {t('factory.summary_processing')}:{' '}
            <span className={`font-bold ${processing > 0 ? 'text-blue-400' : 'text-gray-500'}`}>{processing}</span>
          </span>
          <span className="text-gray-400">
            {t('factory.summary_completed')}:{' '}
            <span className={`font-bold ${completed > 0 ? 'text-green-400' : 'text-gray-500'}`}>{completed}</span>
          </span>
          <span className="text-gray-400">
            {t('factory.summary_failed')}:{' '}
            <span className={`font-bold ${failed > 0 ? 'text-red-400' : 'text-gray-500'}`}>{failed}</span>
          </span>

          {/* Progress bar when there are active jobs */}
          {(pending + processing) > 0 && (completed + pending + processing) > 0 && (
            <div className="flex items-center gap-2 flex-1 max-w-xs">
              <div className="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full transition-all duration-500"
                  style={{ width: `${(completed / (completed + pending + processing)) * 100}%` }}
                />
              </div>
              <span className="text-[11px] text-gray-500 font-mono tabular-nums">
                {completed}/{completed + pending + processing}
              </span>
            </div>
          )}

          <button
            onClick={fetchStats}
            className="ml-auto text-gray-500 hover:text-gray-300 transition-colors"
            title="Refresh"
          >
            <RefreshCw size={12} />
          </button>
        </div>
      </div>

      {/* Sub-tabs */}
      <div className="flex overflow-x-auto border-b border-gray-700 px-4 pt-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap flex-shrink-0 ${
              activeTab === id
                ? 'border-blue-500 text-white'
                : 'border-transparent text-gray-400 hover:text-white'
            }`}
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'pipeline' && (
          <PipelineBlackboard workerProgress={workerProgress} reloadSignal={queueReloadSignal} />
        )}
        {activeTab === 'workers' && (
          <div className="p-4 space-y-6">
            {/* Admin: full workers panel; Non-admin: my workers + connect */}
            {isAdmin ? (
              <WorkersPanel />
            ) : (
              <>
                <ClientWorkerView
                  appMode={appMode}
                  isWorkerRunning={isWorkerRunning}
                  workerProgress={workerProgress}
                  onWorkerStart={onWorkerStart}
                  onWorkerStop={onWorkerStop}
                />
              </>
            )}
          </div>
        )}
        {activeTab === 'dashboard' && (
          <div className="p-4">
            <DashboardPanel />
          </div>
        )}
        {activeTab === 'logs' && (
          <div className="p-4 text-gray-500 text-center">
            <FileText size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-sm">Processing logs will appear here</p>
          </div>
        )}
      </div>
    </div>
  );
}
