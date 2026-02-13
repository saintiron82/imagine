import React, { useState, useEffect } from 'react';
import { FolderPlus, Trash2, RefreshCw, Loader2 } from 'lucide-react';
import { useLocale } from '../i18n';

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
        // Scan first folder; App handles sequential scanning via discoverQueueRef
        onScanFolder(validFolders);
    };

    const handleScanOne = (folderPath) => {
        if (!onScanFolder) return;
        onScanFolder([folderPath]);
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
                                <button
                                    onClick={() => handleScanOne(folder.path)}
                                    className="text-gray-500 hover:text-green-400 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                                    title={t('action.scan_now')}
                                >
                                    <RefreshCw size={12} />
                                </button>
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
        </div>
    );
};

export default RegisteredFoldersPanel;
