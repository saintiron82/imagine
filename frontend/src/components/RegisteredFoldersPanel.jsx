import React, { useState, useEffect } from 'react';
import { FolderPlus, Trash2, RefreshCw, Loader2, FolderSync, ArrowRightLeft, FileX, FilePlus2, Check, X } from 'lucide-react';
import { useLocale } from '../i18n';
import { syncFolder, syncApplyMoves, syncDeleteMissing } from '../services/bridge';

/**
 * Registered Folders panel in Settings.
 *
 * Props:
 *  - onScanFolder(folderPath): triggers App-level discover pipeline
 *  - isBusy: true if pipeline/discover is running (disables scan buttons)
 */
const RegisteredFoldersPanel = ({ onScanFolder, isBusy }) => {
    const { t } = useLocale();
    const [folders, setFolders] = useState([]);
    const [autoScan, setAutoScan] = useState(true);
    const [loading, setLoading] = useState(true);

    // Sync state
    const [syncingFolder, setSyncingFolder] = useState(null);
    const [syncResult, setSyncResult] = useState(null);
    const [syncApplying, setSyncApplying] = useState(false);

    useEffect(() => {
        loadFolders();
    }, []);

    const loadFolders = async () => {
        setLoading(true);
        try {
            const result = await window.electron?.pipeline?.getRegisteredFolders();
            if (result?.success) {
                setFolders(result.folders || []);
                setAutoScan(result.autoScan !== false);
            }
        } catch (e) {
            console.error('Failed to load registered folders:', e);
        } finally {
            setLoading(false);
        }
    };

    const handleAddFolder = async () => {
        try {
            const result = await window.electron?.pipeline?.addRegisteredFolder();
            if (result?.success && result.folders) {
                setFolders(result.folders);
            }
        } catch (e) {
            console.error('Failed to add folder:', e);
        }
    };

    const handleRemoveFolder = async (folderPath) => {
        try {
            const result = await window.electron?.pipeline?.removeRegisteredFolder(folderPath);
            if (result?.success) {
                setFolders(result.folders || []);
            }
        } catch (e) {
            console.error('Failed to remove folder:', e);
        }
    };

    const handleAutoScanToggle = async () => {
        const newValue = !autoScan;
        setAutoScan(newValue);
        try {
            await window.electron?.pipeline?.updateConfig('registered_folders.auto_scan', newValue);
        } catch (e) {
            console.error('Failed to update auto_scan:', e);
            setAutoScan(!newValue);
        }
    };

    const handleScanAll = () => {
        const validFolders = folders.filter(f => f.exists).map(f => f.path);
        if (validFolders.length === 0 || !onScanFolder) return;
        onScanFolder(validFolders);
    };

    const handleScanOne = (folderPath) => {
        if (!onScanFolder) return;
        onScanFolder([folderPath]);
    };

    // ── Sync handlers ────────────────────────────────────────
    const handleSync = async (folderPath) => {
        setSyncingFolder(folderPath);
        setSyncResult(null);
        try {
            const result = await syncFolder(folderPath);
            setSyncResult({ folderPath, ...result });
        } catch (e) {
            console.error('Sync failed:', e);
            setSyncResult({ folderPath, success: false, error: e.message });
        } finally {
            setSyncingFolder(null);
        }
    };

    const handleApplyMoves = async () => {
        if (!syncResult?.moved_list?.length) return;
        setSyncApplying(true);
        try {
            await syncApplyMoves(syncResult.moved_list);
            // Update result to reflect applied moves
            setSyncResult(prev => ({
                ...prev,
                matched: (prev.matched || 0) + (prev.moved || 0),
                moved: 0,
                moved_list: [],
            }));
        } catch (e) {
            console.error('Apply moves failed:', e);
        } finally {
            setSyncApplying(false);
        }
    };

    const handleDeleteMissing = async () => {
        if (!syncResult?.missing_list?.length) return;
        setSyncApplying(true);
        try {
            const ids = syncResult.missing_list.map(f => f.id);
            await syncDeleteMissing(ids);
            setSyncResult(prev => ({
                ...prev,
                missing: 0,
                missing_list: [],
            }));
        } catch (e) {
            console.error('Delete missing failed:', e);
        } finally {
            setSyncApplying(false);
        }
    };

    const handleProcessNew = () => {
        if (!syncResult?.folderPath || !onScanFolder) return;
        onScanFolder([syncResult.folderPath]);
        setSyncResult(null);
    };

    if (loading) {
        return (
            <div className="bg-gray-900/50 rounded border border-gray-700 p-4">
                <div className="flex justify-center p-4"><Loader2 className="animate-spin text-blue-500" size={20} /></div>
            </div>
        );
    }

    return (
        <div className="bg-gray-900/50 rounded border border-gray-700 p-4">
            <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-bold text-gray-400">{t('settings.registered_folders')}</h3>
                <div className="flex gap-2">
                    <button
                        onClick={handleScanAll}
                        disabled={isBusy || folders.filter(f => f.exists).length === 0}
                        className="px-2 py-1 text-xs bg-green-700 hover:bg-green-600 text-white rounded disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1"
                    >
                        {isBusy ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
                        {t('action.scan_all')}
                    </button>
                    <button
                        onClick={handleAddFolder}
                        disabled={isBusy}
                        className="px-2 py-1 text-xs bg-blue-700 hover:bg-blue-600 text-white rounded disabled:opacity-40 flex items-center gap-1"
                    >
                        <FolderPlus size={12} />
                        {t('action.add_folder')}
                    </button>
                </div>
            </div>

            <div className="text-xs text-gray-500 mb-3">{t('settings.registered_folders_desc')}</div>

            {/* Auto-scan toggle */}
            <div className="flex items-center justify-between mb-3 p-2 bg-gray-800/50 rounded">
                <div>
                    <div className="text-sm text-gray-300">{t('settings.auto_scan')}</div>
                    <div className="text-xs text-gray-500">{t('settings.auto_scan_desc')}</div>
                </div>
                <button
                    onClick={handleAutoScanToggle}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 ${
                        autoScan ? 'bg-blue-600' : 'bg-gray-600'
                    }`}
                >
                    <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                            autoScan ? 'translate-x-6' : 'translate-x-1'
                        }`}
                    />
                </button>
            </div>

            {/* Folder list */}
            <div className="space-y-1 max-h-48 overflow-y-auto custom-scrollbar">
                {folders.length === 0 ? (
                    <div className="text-xs text-gray-500 text-center py-4">{t('msg.no_registered_folders')}</div>
                ) : (
                    folders.map((folder) => (
                        <div key={folder.path} className="flex items-center gap-2 p-2 bg-gray-800 rounded group hover:bg-gray-750">
                            <div className={`w-2 h-2 rounded-full flex-shrink-0 ${folder.exists ? 'bg-green-500' : 'bg-red-500'}`} />
                            <span className="text-xs text-gray-300 truncate flex-1" title={folder.path}>{folder.path}</span>
                            {!folder.exists && (
                                <span className="text-xs text-red-400 flex-shrink-0">{t('msg.folder_not_found')}</span>
                            )}
                            {folder.exists && !isBusy && (
                                <>
                                    <button
                                        onClick={() => handleSync(folder.path)}
                                        disabled={syncingFolder === folder.path}
                                        className="text-gray-500 hover:text-yellow-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 disabled:opacity-40"
                                        title={t('action.sync_folder')}
                                    >
                                        {syncingFolder === folder.path ? <Loader2 size={12} className="animate-spin" /> : <FolderSync size={12} />}
                                    </button>
                                    <button
                                        onClick={() => handleScanOne(folder.path)}
                                        className="text-gray-500 hover:text-green-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                                        title={t('action.scan_now')}
                                    >
                                        <RefreshCw size={12} />
                                    </button>
                                </>
                            )}
                            <button
                                onClick={() => handleRemoveFolder(folder.path)}
                                disabled={isBusy}
                                className="text-gray-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 disabled:opacity-40"
                                title={t('action.remove_folder')}
                            >
                                <Trash2 size={12} />
                            </button>
                        </div>
                    ))
                )}
            </div>

            {/* Sync result panel */}
            {syncResult && syncResult.success && (
                <div className="mt-3 p-3 bg-gray-800 rounded border border-gray-600">
                    <div className="flex items-center justify-between mb-2">
                        <h4 className="text-xs font-bold text-gray-300">{t('sync.result_title')}</h4>
                        <button onClick={() => setSyncResult(null)} className="text-gray-500 hover:text-gray-300">
                            <X size={14} />
                        </button>
                    </div>
                    <div className="text-xs text-gray-500 truncate mb-2" title={syncResult.folderPath}>
                        {syncResult.folderPath}
                    </div>

                    {/* Stats grid */}
                    <div className="grid grid-cols-4 gap-2 mb-3">
                        <div className="text-center p-1.5 bg-gray-900 rounded">
                            <div className="text-sm font-bold text-green-400">{syncResult.matched}</div>
                            <div className="text-[10px] text-gray-500">{t('sync.matched')}</div>
                        </div>
                        <div className="text-center p-1.5 bg-gray-900 rounded">
                            <div className={`text-sm font-bold ${syncResult.moved > 0 ? 'text-yellow-400' : 'text-gray-500'}`}>{syncResult.moved}</div>
                            <div className="text-[10px] text-gray-500">{t('sync.moved')}</div>
                        </div>
                        <div className="text-center p-1.5 bg-gray-900 rounded">
                            <div className={`text-sm font-bold ${syncResult.missing > 0 ? 'text-red-400' : 'text-gray-500'}`}>{syncResult.missing}</div>
                            <div className="text-[10px] text-gray-500">{t('sync.missing')}</div>
                        </div>
                        <div className="text-center p-1.5 bg-gray-900 rounded">
                            <div className={`text-sm font-bold ${syncResult.new_files > 0 ? 'text-blue-400' : 'text-gray-500'}`}>{syncResult.new_files}</div>
                            <div className="text-[10px] text-gray-500">{t('sync.new_files')}</div>
                        </div>
                    </div>

                    {/* Action buttons */}
                    <div className="flex flex-wrap gap-2">
                        {syncResult.moved > 0 && (
                            <button
                                onClick={handleApplyMoves}
                                disabled={syncApplying}
                                className="px-2 py-1 text-xs bg-yellow-700 hover:bg-yellow-600 text-white rounded disabled:opacity-40 flex items-center gap-1"
                            >
                                {syncApplying ? <Loader2 size={10} className="animate-spin" /> : <ArrowRightLeft size={10} />}
                                {t('sync.apply_moves')}
                            </button>
                        )}
                        {syncResult.missing > 0 && (
                            <button
                                onClick={handleDeleteMissing}
                                disabled={syncApplying}
                                className="px-2 py-1 text-xs bg-red-700 hover:bg-red-600 text-white rounded disabled:opacity-40 flex items-center gap-1"
                            >
                                {syncApplying ? <Loader2 size={10} className="animate-spin" /> : <FileX size={10} />}
                                {t('sync.delete_missing')}
                            </button>
                        )}
                        {syncResult.new_files > 0 && (
                            <button
                                onClick={handleProcessNew}
                                disabled={isBusy}
                                className="px-2 py-1 text-xs bg-blue-700 hover:bg-blue-600 text-white rounded disabled:opacity-40 flex items-center gap-1"
                            >
                                <FilePlus2 size={10} />
                                {t('sync.process_new')}
                            </button>
                        )}
                        {syncResult.moved === 0 && syncResult.missing === 0 && syncResult.new_files === 0 && (
                            <div className="flex items-center gap-1 text-xs text-green-400">
                                <Check size={12} />
                                {t('sync.all_synced')}
                            </div>
                        )}
                    </div>

                    {/* Moved files detail (collapsible) */}
                    {syncResult.moved_list?.length > 0 && (
                        <details className="mt-2">
                            <summary className="text-[10px] text-gray-500 cursor-pointer hover:text-gray-400">
                                {t('sync.moved_details', { count: syncResult.moved_list.length })}
                            </summary>
                            <div className="mt-1 max-h-24 overflow-y-auto custom-scrollbar space-y-0.5">
                                {syncResult.moved_list.map((m, i) => (
                                    <div key={i} className="text-[10px] text-gray-500 truncate" title={`${m.old_path} → ${m.new_path}`}>
                                        <span className="text-red-400">{m.file_name}</span> → <span className="text-green-400">{m.new_path.split('/').pop()}</span>
                                    </div>
                                ))}
                            </div>
                        </details>
                    )}

                    {/* Missing files detail */}
                    {syncResult.missing_list?.length > 0 && (
                        <details className="mt-2">
                            <summary className="text-[10px] text-gray-500 cursor-pointer hover:text-gray-400">
                                {t('sync.missing_details', { count: syncResult.missing_list.length })}
                            </summary>
                            <div className="mt-1 max-h-24 overflow-y-auto custom-scrollbar space-y-0.5">
                                {syncResult.missing_list.map((m, i) => (
                                    <div key={i} className="text-[10px] text-red-400 truncate" title={m.file_path}>
                                        {m.file_name}
                                    </div>
                                ))}
                            </div>
                        </details>
                    )}
                </div>
            )}

            {/* Sync error */}
            {syncResult && !syncResult.success && (
                <div className="mt-3 p-2 bg-red-900/30 rounded border border-red-700 text-xs text-red-400">
                    {t('sync.error')}: {syncResult.error}
                    <button onClick={() => setSyncResult(null)} className="ml-2 text-red-300 hover:text-white">
                        <X size={12} className="inline" />
                    </button>
                </div>
            )}
        </div>
    );
};

export default RegisteredFoldersPanel;
