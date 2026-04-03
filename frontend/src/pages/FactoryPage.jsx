/**
 * FactoryPage — Processing factory: Pipeline dashboard + Workers + Dashboard + Logs
 */

import { useState, useEffect, useCallback } from 'react';
import { Activity, Server, BarChart3, FileText, RefreshCw } from 'lucide-react';
import { useLocale } from '../i18n';
import { isElectron } from '../api/client';
import { getJobStats } from '../api/worker';
import { WorkersPanel, DashboardPanel } from './AdminPage';
import WorkerPage, { MyWorkersSection, ConnectMyPC } from './WorkerPage';


export default function FactoryPage({
  isAdmin,
  appMode,
  queueReloadSignal,
}) {
  const { t } = useLocale();
  const [activeTab, setActiveTab] = useState('workers');
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

  const allTabs = [
    { id: 'workers', label: t('factory.tab_workers'), icon: Server, adminOnly: false },
    { id: 'dashboard', label: t('factory.tab_dashboard'), icon: BarChart3, adminOnly: true },
    { id: 'logs', label: t('factory.tab_logs'), icon: FileText, adminOnly: true },
  ];
  const tabs = isAdmin ? allTabs : allTabs.filter(tab => !tab.adminOnly);

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      {/* Legacy summary bar removed — WorkersPanel has unified dashboard */}

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
        {activeTab === 'workers' && (
          isAdmin ? (
            <div className="p-4 space-y-6">
              <WorkersPanel />
            </div>
          ) : (
            <WorkerPage appMode="client" />
          )
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
