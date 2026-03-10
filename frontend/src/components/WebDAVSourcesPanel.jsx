import React, { useState, useEffect, useCallback } from 'react';
import { Globe, Trash2, RefreshCw, Loader2, Download, Plus } from 'lucide-react';
import { useLocale } from '../i18n';
import WebDAVConnectDialog from './WebDAVConnectDialog';

/**
 * WebDAV Sources panel — shown alongside RegisteredFoldersPanel in Settings.
 *
 * Props:
 *  - onScanFolder([path]): triggers App-level discover pipeline on cache folder
 *  - isBusy: true if pipeline/discover is running
 */
const WebDAVSourcesPanel = ({ onScanFolder, isBusy }) => {
    const { t } = useLocale();
    const [sources, setSources] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showAddDialog, setShowAddDialog] = useState(false);
    const [syncingSource, setSyncingSource] = useState(null);
    const [syncProgress, setSyncProgress] = useState(null);

    const loadSources = useCallback(async () => {
        setLoading(true);
        try {
            const result = await window.electron?.webdav?.getSources();
            if (result?.success) {
                setSources(result.sources || []);
            }
        } catch (e) {
            console.error('Failed to load WebDAV sources:', e);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        loadSources();
    }, [loadSources]);

    // Listen to sync events
    useEffect(() => {
        if (!window.electron?.webdav) return;

        const handleProgress = (data) => {
            setSyncProgress(data);
        };

        const handleComplete = (data) => {
            setSyncingSource(null);
            setSyncProgress(null);
            loadSources(); // Refresh last_sync times

            // Auto-discover after successful sync
            if (data?.sourceId && !data?.error) {
                handleDiscoverAfterSync(data.sourceId);
            }
        };

        window.electron.webdav.onSyncProgress(handleProgress);
        window.electron.webdav.onSyncComplete(handleComplete);

        return () => {
            window.electron.webdav.offSyncProgress();
            window.electron.webdav.offSyncComplete();
        };
    }, [loadSources]);

    const handleDiscoverAfterSync = async (sourceId) => {
        if (!onScanFolder) return;
        try {
            const cachePath = await window.electron?.webdav?.getCachePath(sourceId);
            if (cachePath) {
                onScanFolder([cachePath]);
            }
        } catch (e) {
            console.error('Failed to get cache path:', e);
        }
    };

    const handleSync = (sourceId) => {
        if (syncingSource) return;
        setSyncingSource(sourceId);
        setSyncProgress(null);
        window.electron?.webdav?.syncSource(sourceId);
    };

    const handleRemove = async (sourceId) => {
        try {
            const result = await window.electron?.webdav?.removeSource(sourceId);
            if (result?.success) {
                setSources(prev => prev.filter(s => s.id !== sourceId));
            }
        } catch (e) {
            console.error('Failed to remove WebDAV source:', e);
        }
    };

    const handleSourceAdded = () => {
        loadSources();
    };

    const formatLastSync = (iso) => {
        if (!iso) return t('webdav.never_synced');
        try {
            const d = new Date(iso);
            const now = new Date();
            const diffMs = now - d;
            const diffMin = Math.floor(diffMs / 60000);
            if (diffMin < 1) return t('webdav.just_now');
            if (diffMin < 60) return `${diffMin}m ago`;
            const diffHr = Math.floor(diffMin / 60);
            if (diffHr < 24) return `${diffHr}h ago`;
            return d.toLocaleDateString();
        } catch {
            return iso;
        }
    };

    if (loading) {
        return (
            <div className="bg-gray-900/50 rounded border border-gray-700 p-4 mt-3">
                <div className="flex justify-center p-4"><Loader2 className="animate-spin text-blue-500" size={20} /></div>
            </div>
        );
    }

    return (
        <div className="bg-gray-900/50 rounded border border-gray-700 p-4 mt-3">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-gray-400 flex items-center gap-1.5">
                    <Globe size={14} />
                    {t('webdav.title')}
                </h3>
                <button
                    onClick={() => setShowAddDialog(true)}
                    disabled={isBusy}
                    className="px-2 py-1 text-xs bg-blue-700 hover:bg-blue-600 text-white rounded disabled:opacity-40 flex items-center gap-1"
                >
                    <Plus size={12} />
                    {t('webdav.add')}
                </button>
            </div>

            <div className="text-xs text-gray-500 mb-3">{t('webdav.desc')}</div>

            {/* Source list */}
            <div className="space-y-1 max-h-48 overflow-y-auto custom-scrollbar">
                {sources.length === 0 ? (
                    <div className="text-xs text-gray-500 text-center py-4">{t('webdav.no_sources')}</div>
                ) : (
                    sources.map((source) => (
                        <div key={source.id} className="p-2 bg-gray-800 rounded group hover:bg-gray-750">
                            <div className="flex items-center gap-2">
                                <Globe size={12} className="text-blue-400 flex-shrink-0" />
                                <div className="flex-1 min-w-0">
                                    <div className="text-xs text-gray-300 truncate font-medium">
                                        {source.name || source.url}
                                    </div>
                                    <div className="text-[10px] text-gray-500 truncate">
                                        {source.url}{source.remote_path !== '/' ? source.remote_path : ''}
                                    </div>
                                </div>
                                <div className="text-[10px] text-gray-500 flex-shrink-0">
                                    {formatLastSync(source.last_sync)}
                                </div>

                                {/* Sync button */}
                                <button
                                    onClick={() => handleSync(source.id)}
                                    disabled={isBusy || syncingSource === source.id}
                                    className="text-gray-500 hover:text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 disabled:opacity-40"
                                    title={t('webdav.sync_and_scan')}
                                >
                                    {syncingSource === source.id
                                        ? <Loader2 size={12} className="animate-spin" />
                                        : <Download size={12} />
                                    }
                                </button>

                                {/* Remove button */}
                                <button
                                    onClick={() => handleRemove(source.id)}
                                    disabled={isBusy || syncingSource === source.id}
                                    className="text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 disabled:opacity-40"
                                    title={t('webdav.remove')}
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>

                            {/* Sync progress (when syncing this source) */}
                            {syncingSource === source.id && syncProgress && (
                                <div className="mt-2 text-[10px] text-blue-400">
                                    {syncProgress.event === 'listing' && t('webdav.listing')}
                                    {syncProgress.event === 'comparing' && t('webdav.comparing')}
                                    {syncProgress.event === 'downloading' && (
                                        <div className="flex items-center gap-2">
                                            <span>{t('webdav.sync_progress', {
                                                index: syncProgress.index,
                                                total: syncProgress.total,
                                            })}</span>
                                            <div className="flex-1 h-1 bg-gray-700 rounded overflow-hidden">
                                                <div
                                                    className="h-full bg-blue-500 transition-all"
                                                    style={{ width: `${(syncProgress.index / syncProgress.total) * 100}%` }}
                                                />
                                            </div>
                                        </div>
                                    )}
                                    {syncProgress.event === 'cleaning' && syncProgress.message}
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>

            {/* Add dialog */}
            {showAddDialog && (
                <WebDAVConnectDialog
                    onClose={() => setShowAddDialog(false)}
                    onAdded={handleSourceAdded}
                />
            )}
        </div>
    );
};

export default WebDAVSourcesPanel;
