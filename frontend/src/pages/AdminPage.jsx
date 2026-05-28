/**
 * AdminPage — tab router for admin panels.
 * All panel components are in frontend/src/components/admin/.
 */

import { useState } from 'react';
import { useLocale } from '../i18n';
import { useAuth } from '../contexts/AuthContext';
import { UserCheck, Tag, Clock, Wrench, Network, MessageSquare } from 'lucide-react';

import MembersPanel from '../components/admin/MembersPanel';
import ClassificationPanel from '../components/admin/ClassificationPanel';
import HistoryPanel from '../components/admin/HistoryPanel';
import ToolsPanel from '../components/admin/ToolsPanel';
import ConnectionInfoPanel from '../components/admin/ConnectionInfoPanel';
import SearchFeedbackPanel from '../components/admin/SearchFeedbackPanel';

// Re-export panels used by other pages (e.g. FactoryPage)
export { default as WorkersPanel } from '../components/admin/WorkersPanel';
export { default as DashboardPanel } from '../components/admin/DashboardPanel';
export { default as DiscoverPanel } from '../components/admin/DiscoverPanel';

export default function AdminPage() {
  const { t } = useLocale();
  const { user: currentUser } = useAuth();
  const [activeTab, setActiveTab] = useState('members');

  const tabs = [
    { id: 'members', label: t('admin.tab_members'), icon: UserCheck },
    { id: 'classification', label: t('admin.tab_classification'), icon: Tag },
    { id: 'history', label: t('admin.tab_history'), icon: Clock },
    { id: 'tools', label: t('admin.tab_tools'), icon: Wrench },
    { id: 'connection', label: t('admin.tab_connection'), icon: Network },
    { id: 'feedback', label: t('admin.tab_search_feedback'), icon: MessageSquare },
  ];

  return (
    <div className="flex flex-col h-full bg-gray-900 text-white">
      {/* Tab bar */}
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
      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'members' && <MembersPanel currentUser={currentUser} />}
        {activeTab === 'classification' && <ClassificationPanel />}
        {activeTab === 'history' && <HistoryPanel />}
        {activeTab === 'tools' && <ToolsPanel />}
        {activeTab === 'connection' && <ConnectionInfoPanel />}
        {activeTab === 'feedback' && <SearchFeedbackPanel />}
      </div>
    </div>
  );
}
