import React, { useState, useEffect, useCallback } from 'react';
import { Globe, Trash2, Play, Loader2, Plus } from 'lucide-react';
import { useLocale } from '../i18n';
import WebDAVConnectDialog from './WebDAVConnectDialog';

/**
 * WebDAV Sources panel — shown in Archive tab sidebar.
 * Fetch-and-Process: files stay on NAS, pipeline processes them via FileContainer.
 *
 * Props:
 *  - isBusy: true if pipeline/discover is running
 */
const WebDAVSourcesPanel = ({ isBusy }) => {
    const { t } = useLocale();
    const [sources, setSources] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showAddDialog, setShowAddDialog] = useState(false);
    const [processingSrc, setProcessingSrc] = useState(null);
    const [progress, setProgress] = useState(null);

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

    // Listen to process events
    useEffect(() => {
        if (!window.electron?.webdav) return;

        const handleProgress = (data) => {
            setProgress(data);
        };

        const handleComplete = (data) => {
            setProcessingSrc(null);
            setProgress(null);
            loadSources();
        };

        window.electron.webdav.onSyncProgress(handleProgress);
        window.electron.webdav.onSyncComplete(handleComplete);

        return () => {
            window.electron.webdav.offSyncProgress();
            window.electron.webdav.offSyncComplete();
        };
    }, [loadSources]);

    const handleProcess = (sourceId) => {
        if (processingSrc) return;
        setProcessingSrc(sourceId);
        setProgress(null);
        window.electron?.webdav?.processSource(sourceId);
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

    const formatLastScan = (iso) => {
        if (!iso) return t('webdav.never_scanned');
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
                                    {formatLastScan(source.last_scan || source.last_sync)}
                                </div>

                                {/* Scan button */}
                                <button
                                    onClick={() => handleProcess(source.id)}
                                    disabled={isBusy || processingSrc === source.id}
                                    className="text-gray-500 hover:text-green-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 disabled:opacity-40"
                                    title={t('webdav.scan')}
                                >
                                    {processingSrc === source.id
                                        ? <Loader2 size={12} className="animate-spin" />
                                        : <Play size={12} />
                                    }
                                </button>

                                {/* Remove button */}
                                <button
                                    onClick={() => handleRemove(source.id)}
                                    disabled={isBusy || processingSrc === source.id}
                                    className="text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 disabled:opacity-40"
                                    title={t('webdav.remove')}
                                >
                                    <Trash2 size={12} />
                                </button>
                            </div>

                            {/* Processing progress */}
                            {processingSrc === source.id && progress && (
                                <div className="mt-2 text-[10px] text-blue-400 truncate">
                                    {progress.message || 'Processing...'}
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
