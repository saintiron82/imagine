/**
 * FactoryPage — Processing factory: Queue dashboard + Workers + Dashboard + Logs
 * Phase 1: Placeholder shell (to be expanded in Phase 3)
 */

import { useState } from 'react';
import { Activity, Server, BarChart3, FileText } from 'lucide-react';
import { useLocale } from '../i18n';

export default function FactoryPage({ isAdmin }) {
  const { t } = useLocale();
  const [activeTab, setActiveTab] = useState('queue');

  const tabs = [
    { id: 'queue', label: t('factory.tab_queue'), icon: Activity },
    { id: 'workers', label: t('factory.tab_workers'), icon: Server },
    { id: 'dashboard', label: t('factory.tab_dashboard'), icon: BarChart3 },
    { id: 'logs', label: t('factory.tab_logs'), icon: FileText },
  ];

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      {/* Summary bar */}
      <div className="px-4 py-3 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-4 text-sm">
          <span className="text-gray-400">{t('factory.summary_pending')}: <span className="text-yellow-400 font-medium">—</span></span>
          <span className="text-gray-400">{t('factory.summary_processing')}: <span className="text-blue-400 font-medium">—</span></span>
          <span className="text-gray-400">{t('factory.summary_completed')}: <span className="text-green-400 font-medium">—</span></span>
          <span className="text-gray-400">{t('factory.summary_failed')}: <span className="text-red-400 font-medium">—</span></span>
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

      {/* Content placeholder */}
      <div className="flex-1 overflow-auto p-6">
        <div className="flex items-center justify-center h-full text-gray-500">
          <div className="text-center">
            <Activity size={48} className="mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">{tabs.find(t => t.id === activeTab)?.label}</p>
            <p className="text-sm mt-1">Coming in Phase 3</p>
          </div>
        </div>
      </div>
    </div>
  );
}
