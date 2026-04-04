/**
 * DashboardPanel — DB stats overview (저장소 정보).
 * Download cache UI removed — downloads are queue-lifecycle managed.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import { RefreshCw, Loader2 } from 'lucide-react';


export default function DashboardPanel() {
  const { t } = useLocale();
  const [dbStats, setDbStats] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchAll = useCallback(async () => {
    try {
      setLoading(true);
      const dbResult = await window.electron?.pipeline?.getDbStats?.().catch(() => null);
      if (dbResult) setDbStats(dbResult);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  if (loading && !dbStats) {
    return <div className="flex items-center gap-2 text-gray-400"><Loader2 className="animate-spin" size={16} /> Loading...</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{t('admin.tab_dashboard') || 'Storage'}</h2>
        <button onClick={fetchAll} className="p-1.5 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition-colors">
          <RefreshCw size={16} />
        </button>
      </div>

      {/* DB Stats */}
      {dbStats && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-gray-300 mb-3">{t('admin.dash_db_stats') || 'DB Stats'}</h3>
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-white font-mono">{(dbStats.total ?? 0).toLocaleString()}</div>
              <div className="text-xs text-gray-400">{t('admin.dash_total') || 'Total'}</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400 font-mono">{(dbStats.searchable ?? 0).toLocaleString()}</div>
              <div className="text-xs text-gray-400">{t('admin.dash_searchable') || 'Searchable'}</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-amber-400 font-mono">{(dbStats.preview_only ?? 0).toLocaleString()}</div>
              <div className="text-xs text-gray-400">{t('admin.dash_preview') || 'Preview Only'}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
