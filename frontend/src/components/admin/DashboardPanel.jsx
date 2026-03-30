/**
 * DashboardPanel — DB stats + download cache overview.
 * Extracted from AdminPage.jsx.
 */

import { useState, useEffect, useCallback } from 'react';
import { useLocale } from '../../i18n';
import { RefreshCw, Loader2, Trash2 } from 'lucide-react';


export default function DashboardPanel() {
  const { t } = useLocale();
  const [cacheData, setCacheData] = useState(null);
  const [dbStats, setDbStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [cleaning, setCleaning] = useState(false);

  const fetchAll = useCallback(async () => {
    try {
      setLoading(true);
      const [cacheResult, dbResult] = await Promise.all([
        window.electron?.pipeline?.downloadCache?.stats?.().catch(() => null),
        window.electron?.pipeline?.getDbStats?.().catch(() => null),
      ]);
      if (cacheResult) setCacheData(cacheResult);
      if (dbResult) setDbStats(dbResult);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const handleCacheCleanup = async () => {
    setCleaning(true);
    try {
      const result = await window.electron?.pipeline?.downloadCache?.cleanup?.();
      if (result) {
        alert(`${result.deleted} files deleted (${result.freed_mb} MB freed)`);
        fetchAll();
      }
    } finally {
      setCleaning(false);
    }
  };

  if (loading && !cacheData) {
    return <div className="flex items-center gap-2 text-gray-400"><Loader2 className="animate-spin" size={16} /> Loading...</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">{t('admin.tab_dashboard') || 'Dashboard'}</h2>
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

      {/* Download Cache */}
      {cacheData && (
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-gray-300">{t('admin.dash_cache') || 'Download Cache'}</h3>
            <button
              onClick={handleCacheCleanup}
              disabled={cleaning || cacheData.file_count === 0}
              className="px-2 py-1 text-xs bg-red-900/30 hover:bg-red-800/50 text-red-400 rounded disabled:opacity-30 disabled:cursor-not-allowed"
            >
              {cleaning ? <Loader2 size={12} className="animate-spin inline" /> : <Trash2 size={12} className="inline" />}
              {' '}{t('admin.dash_cleanup') || 'Clean Up'}
            </button>
          </div>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-sm text-gray-400">{cacheData.file_count.toLocaleString()} files</span>
            <span className="text-gray-600">|</span>
            <span className="text-sm font-mono text-white">
              {cacheData.total_mb >= 1024
                ? `${(cacheData.total_mb / 1024).toFixed(1)} GB`
                : `${cacheData.total_mb} MB`}
            </span>
            {cacheData.limit_gb > 0 && (
              <>
                <span className="text-gray-600">/</span>
                <span className="text-sm text-gray-400">{cacheData.limit_gb} GB</span>
              </>
            )}
          </div>
          {cacheData.limit_gb > 0 && (
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  cacheData.total_mb / 1024 / cacheData.limit_gb > 0.8 ? 'bg-amber-500' : 'bg-blue-500'
                }`}
                style={{ width: `${Math.min(100, (cacheData.total_mb / 1024 / cacheData.limit_gb) * 100)}%` }}
              />
            </div>
          )}
        </div>
      )}

    </div>
  );
}
