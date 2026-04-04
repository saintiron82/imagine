/**
 * AnalysisPage — Top-level "분석" tab.
 *
 * Sub-tabs:
 *   - 대시보드: Queue progress + worker overview (FactoryPage/WorkersPanel)
 *   - 워커: My workers, connect PC, worker settings (WorkerPage)
 */

import { useState } from 'react';
import { BarChart3, Cpu } from 'lucide-react';
import { useLocale } from '../i18n';
import FactoryPage from './FactoryPage';
import WorkerPage from './WorkerPage';


export default function AnalysisPage({ isAdmin, appMode, queueReloadSignal }) {
  const { t } = useLocale();
  const [activeTab, setActiveTab] = useState('dashboard');

  const tabs = [
    { id: 'dashboard', label: t('tab.factory'), icon: BarChart3 },
    { id: 'worker', label: t('tab.worker'), icon: Cpu },
  ];

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      {/* Sub-tabs */}
      <div className="flex overflow-x-auto border-b border-gray-700 px-4 pt-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 px-4 py-2 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === id
                ? 'border-emerald-500 text-emerald-400'
                : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-500'
            }`}
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'dashboard' ? (
          <FactoryPage
            isAdmin={isAdmin}
            appMode={appMode}
            queueReloadSignal={queueReloadSignal}
          />
        ) : activeTab === 'worker' ? (
          <WorkerPage />
        ) : null}
      </div>
    </div>
  );
}
